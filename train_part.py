import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import jittor as jt
from jittor import init
from jittor import optim
from jittor.dataset import Dataset
import utils.util as util
import utils.config as config
import h5py

# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.

import jittor.nn as nn


class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=False):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def execute(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = jt.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return jt.concat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class VoxelEncoder(nn.Module): 

    def __init__(self, feat_len, hidden_size=32):
        super(VoxelEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, hidden_size, 4, padding=1, stride=1, bias=False)
        self.in1 = nn.InstanceNorm3d(hidden_size)
        self.conv2 = nn.Conv3d(hidden_size, hidden_size * 2, 4, padding=1, stride=2, bias=False)
        self.in2 = nn.InstanceNorm3d(hidden_size * 2)
        self.conv3 = nn.Conv3d(hidden_size * 2, hidden_size * 4, 4, padding=1, stride=2, bias=False)
        self.in3 = nn.InstanceNorm3d(hidden_size * 4)
        self.conv4 = nn.Conv3d(hidden_size * 4, hidden_size * 8, 4, padding=1, stride=2, bias=False)
        self.in4 = nn.InstanceNorm3d(hidden_size * 8)
        self.conv5 = nn.Conv3d(hidden_size * 8, feat_len, 3, padding=0, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.02)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        init.xavier_uniform_(self.conv3.weight)
        init.xavier_uniform_(self.conv4.weight)
        init.constant_(self.conv5.bias, 0)

    def execute(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, 32, 32, 32)
        x = self.leaky_relu(self.in1(self.conv1(x)))
        x = self.leaky_relu(self.in2(self.conv2(x)))
        x = self.leaky_relu(self.in3(self.conv3(x)))
        x = self.leaky_relu(self.in4(self.conv4(x)))
        x = self.conv5(x).reshape(batch_size, -1)
        return x


class PartEncoder(nn.Module): 

    def __init__(self, feat_len, latent_size, probabilistic=False):
        super(PartEncoder, self).__init__()
        self.vox_enc = VoxelEncoder(feat_len)
        self.mlp1 = nn.Linear(feat_len + 3, latent_size)
        init.gauss_(self.mlp1.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.sampler = PartFeatSampler(latent_size) if probabilistic else None

    def execute(self, x, norms):
        # batch_size = x.shape[0]
        # print(x.shape)
        feat = self.leaky_relu(self.vox_enc(x))
        x = self.mlp1(jt.concat([feat, norms], -1))
        if self.sampler is not None:
            x = self.sampler(x)
        return x

###############################################################################
# Decoder
###############################################################################

class IM_Tiny(nn.Module):
    def __init__(self, feat_len, hidden_size=32):
        super(IM_Tiny, self).__init__()
        self.mlp1 = nn.Linear(feat_len + 3, hidden_size * 8)
        self.mlp2 = nn.Linear(hidden_size * 8, hidden_size * 8)
        self.mlp3 = nn.Linear(hidden_size * 8, hidden_size * 8)
        self.mlp4 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.mlp5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.mlp6 = nn.Linear(hidden_size * 2, hidden_size)
        self.mlp7 = nn.Linear(hidden_size, 1)
        init.gauss_(self.mlp1.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp2.bias, 0)
        init.gauss_(self.mlp3.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp3.bias, 0)
        init.gauss_(self.mlp4.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp4.bias, 0)
        init.gauss_(self.mlp5.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp5.bias, 0)
        init.gauss_(self.mlp6.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp6.bias, 0)
        init.gauss_(self.mlp7.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp7.bias, 0.5)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.sigmoid = nn.Sigmoid()

    def execute(self, net):
        x = self.leaky_relu(self.mlp1(net))
        x = self.leaky_relu(self.mlp2(x))
        x = self.leaky_relu(self.mlp3(x))
        x = self.leaky_relu(self.mlp4(x))
        x = self.leaky_relu(self.mlp5(x))
        x = self.leaky_relu(self.mlp6(x))
        pred = self.sigmoid(self.mlp7(x))

        return pred


class NodeClassifier(nn.Module):
    def __init__(self, feat_len):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(feat_len, 8)
        self.sigmoid = nn.Sigmoid()
        
    def execute(self, x):
        # x = self.leaky_relu(self.mlp1(x))
        x = self.sigmoid(self.mlp1(x))
        return x




class PartDecoder(nn.Module):
    def __init__(self, feat_len):
        super(PartDecoder, self).__init__()
        self.predictor = IM_Tiny(feat_len)
        self.classifier = NodeClassifier(feat_len)
        self.bce_loss = nn.BCELoss()
        
    def execute(self, x, in_feat):
        batch_size, num_points, _ = x.shape
        # node_type = self.classifier(in_feat)
        feat = in_feat.view(batch_size, 1, -1).expand(-1, num_points, -1)
        query = jt.concat([feat, x], -1).view(batch_size * num_points, -1)
        pred = self.predictor(query)
        return pred

    def loss(self, pred, gt):
        bce_loss = self.bce_loss(pred, gt)
        return bce_loss



###############################################################################
# Dataset
###############################################################################

class PartNetGeoDataset(Dataset):
    def __init__(self, conf, split='train', batch_size=1, shuffle=False):
        super(PartNetGeoDataset, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_path = conf.data_path
        if split == 'train':
            item_list = conf.train_list
            self.data_h5 = conf.train_dataset
        elif split == 'val':
            item_list = conf.val_list
            self.data_h5 = conf.val_dataset
        elif split == 'test':
            item_list = conf.test_list
            self.data_h5 = conf.test_dataset
        self.keys = ['cells', 'points', 'values', 'normals', 'voxels']
        self.data_dict = self.load_h5f(conf.load_ram)

    def load_h5f(self, load_ram=True):
        src_file = os.path.join(self.data_path, self.data_h5)
        print(src_file)
        h5f = h5py.File(src_file, 'r')
        return {
            key: np.array(h5f.get(key)) for key in self.keys
        }

    def __getitem__(self, index):
        # model_name = self.items_lst[index]
        # idx = slice(self.idx_end[index], self.idx_end[index + 1])
        cell = self.data_dict['cells'][index]
        normals = self.data_dict['normals'][index]
        voxels = self.data_dict['voxels'][index]
        points = self.data_dict['points'][index]
        values = self.data_dict['values'][index]
        num_points = points.shape[0]
        half = num_points // 2
        pos = np.nonzero(values>0.5)[0]
        neg = np.nonzero(values<0.5)[0]
        resample = np.zeros(num_points, dtype=np.long)
        resample[:half] = pos[np.random.randint(pos.shape[0],size=(half,))].reshape(-1)
        resample[half:] = neg[np.random.randint(neg.shape[0],size=(half,))].reshape(-1)
        points = points[resample]
        values = values[resample]
        return cell, points, values, normals, voxels

    def __len__(self):
        return self.data_dict['cells'].shape[0]

def train(conf):
    # load network model

    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.log_path, conf.exp_name))
    if os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.model_path, conf.exp_name))

    # create directories for this run
    os.makedirs(os.path.join(conf.model_path, conf.exp_name))
    os.makedirs(os.path.join(conf.log_path, conf.exp_name))

    # file log
    flog = open(os.path.join(conf.log_path, conf.exp_name, 'train.log'), 'w')


    # log the object category information
    print(f'Object Category: {conf.category}')
    flog.write(f'Object Category: {conf.category}\n')

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (conf.seed))
    flog.write(f'Random Seed: {conf.seed}\n')
    random.seed(conf.seed)
    np.random.seed(conf.seed)

    # create models
    encoder = PartEncoder(feat_len=conf.geo_feat_size, latent_size=conf.geo_feat_size)
    decoder = PartDecoder(feat_len=conf.geo_feat_size)
    models = [encoder, decoder]
    model_names = ['part_pc_encoder', 'part_pc_decoder']

    # create optimizers
    optimizer = nn.Adam(encoder.parameters() + decoder.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # create training and validation datasets and data loaders
    train_dataloader = PartNetGeoDataset(conf, 'train', batch_size=conf.batch_size, shuffle=True)
    train_num_batch = len(train_dataloader)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch    Dataset    Iteration    Progress(%)     LR      ReconLoss  KLDivLoss  TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pyjt
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'train'))

    # save config
    jt.save(conf, os.path.join(conf.model_path, conf.exp_name, 'conf.pth'))


    # start training
    print("Starting training ...... ")
    flog.write('Starting training ......\n')

    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step = None

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            print(f'training run {conf.exp_name}')
            flog.write(f'training run {conf.exp_name}\n')
            print(header)
            flog.write(header+'\n')


        train_batches = enumerate(train_dataloader, 0)
        train_fraction_done = 0.0
        train_batch_ind = 0

        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, encoder=encoder, decoder=decoder, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind,
                num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, flog=flog)

            # optimize one step
            optimizer.step(total_loss)

            # save checkpoint
            with jt.no_grad():
                if last_checkpoint_step is None or \
                        train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    print("Saving checkpoint ...... ", end='', flush=True)
                    flog.write("Saving checkpoint ...... ")
                    util.save_checkpoint(
                        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
                        epoch=epoch, prepend_epoch=True, optimizers=[optimizer], optimizer_names=['opt'])
                    print("DONE")
                    flog.write("DONE\n")
                    last_checkpoint_step = train_step

    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write('Saving final checkpoint ...... ')
    util.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
        epoch=epoch, prepend_epoch=False, optimizers=[optimizer], optimizer_names=['opt'])
    print("DONE")
    flog.write("DONE\n")

    flog.close()


def forward(batch, encoder, decoder, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, flog=None):
    
    data = [item for item in batch]
    cells, points, values, normals, voxels = data
    batch_size = cells.shape[0]
    

    feat = encoder(voxels, normals)
    
    num_smp = points.shape[1]

    # points = jt.reshape(points, (batch_size * num_smp, -1))
    # values = jt.reshape(values, (batch_size * num_smp, -1))
    
    pred = decoder(points, feat)
    # node_type, pred = decoder(points, feat)
    recon_loss= decoder.loss(pred, values.view(-1, 1))
    recon_loss = recon_loss.mean() * conf.loss_weight_geo
    # mask = gt_type.max(1).values.view(-1,1)
    total_loss = recon_loss
    with jt.no_grad():
        # log to console
        if log_console:
            print(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}% '''
                f'''{recon_loss.item():>11.6f} '''
                f'''{total_loss.item():>10.6f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}% '''
                f'''{recon_loss.item():>11.6f} '''
                f'''{total_loss.item():>10.6f}\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('loss', total_loss.item(), step)
            tb_writer.add_scalar('recon_loss', recon_loss.item(), step)

    return total_loss


if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = config.add_train_vae_args(parser)
    parser.add_argument('--use_local_frame', action='store_true', default=False, help='factorize out 3-dim center + 1-dim scale')
    config = parser.parse_args()

    train(conf=config)

