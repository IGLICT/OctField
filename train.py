
import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
from utils.config import add_train_vae_args
from data.data import ShapeDataset, Tree
import utils.util as util


def train(conf):
    # load network model
    models = util.get_model_module(conf.model_version)

    # check if training run already exists. If so, delete it.
    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)) or \
            os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
        response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
        if response != 'y':
            sys.exit()
    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.log_path, conf.exp_name))
    if os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.model_path, conf.exp_name))

    # create directories for this run
    os.makedirs(os.path.join(conf.model_path, conf.exp_name))
    os.makedirs(os.path.join(conf.log_path, conf.exp_name))

    # file log
    flog = open(os.path.join(conf.log_path, conf.exp_name, 'train.log'), 'w')
    
    jt.use_cuda = True  

    # log the object category information
    print(f'Object Category: {conf.category}')
    flog.write(f'Object Category: {conf.category}\n')

    # save config
    jt.save(conf, os.path.join(conf.model_path, conf.exp_name, 'conf.pth'))

    # create models
    # encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
    encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
    decoder = models.RecursiveDecoder(conf)
    models = [encoder, decoder]
    model_names = ['encoder', 'decoder']

    # load pretrained part AE/VAE
    pretrain_ckpt_dir = os.path.join(conf.model_path, conf.part_pc_exp_name)
    pretrain_ckpt_epoch = conf.part_pc_model_epoch
    print(f'Loading ckpt from {pretrain_ckpt_dir}: epoch {pretrain_ckpt_epoch}')
    __ = util.load_checkpoint(
        # models=[encoder.node_encoder.part_encoder, decoder.part_decoder],
        # model_names=['part_pc_encoder', 'part_pc_decoder'],
        models=[encoder.part_encoder],
        model_names=['part_pc_encoder'],
        dirname=pretrain_ckpt_dir,
        epoch=pretrain_ckpt_epoch,
        strict=True)

    # set part_encoder and part_decoder BatchNorm to eval mode
    encoder.part_encoder.eval()
    for param in encoder.part_encoder.parameters():
        param.requires_grad = False
    # decoder.part_decoder.eval()
    # for param in decoder.part_decoder.parameters():
    #     param.requires_grad = False

    # create optimizers
    optimizer = nn.Adam([encoder.parameters(), decoder.parameters()], lr=conf.lr)

    # create training and validation datasets and data loaders
    train_dataset = ShapeDataset(conf, 'train')
    valdt_dataset = ShapeDataset(conf, 'val')
    train_dataloader = jt.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=lambda x: x)
    valdt_dataloader = jt.utils.data.DataLoader(valdt_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=lambda x: x)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)      LR       LatentLoss  TypeLoss   KLDivLoss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pyjt
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'train'))
        valdt_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'val'))


    # start training
    print("Starting training ...... ")
    flog.write('Starting training ......\n')

    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_valdt_console_log_step = None, None
    train_num_batch, valdt_num_batch = len(train_dataloader), len(valdt_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            print(f'training run {conf.exp_name}')
            flog.write(f'training run {conf.exp_name}\n')
            print(header)
            flog.write(header+'\n')

        train_batches = enumerate(train_dataloader, 0)
        valdt_batches = enumerate(valdt_dataloader, 0)

        train_fraction_done, valdt_fraction_done = 0.0, 0.0
        valdt_batch_ind = -1
        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # make sure the models are in eval mode to deactivate BatchNorm for PartEncoder and PartDecoder
            # there are no other BatchNorm / Dropout in the rest of the network
            for m in models:
                m.eval()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, encoder=encoder, decoder=decoder, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer,
                lr=optimizer.param_groups[0].get('lr'), flog=flog)

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
                        epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    print("DONE")
                    flog.write("DONE\n")
                    last_checkpoint_step = train_step

            # validate one batch
            while valdt_fraction_done <= train_fraction_done and valdt_batch_ind+1 < valdt_num_batch:
                valdt_batch_ind, batch = next(valdt_batches)

                valdt_fraction_done = (valdt_batch_ind + 1) / valdt_num_batch
                valdt_step = (epoch + valdt_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_valdt_console_log_step is None or \
                        valdt_step - last_valdt_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_valdt_console_log_step = valdt_step

                # set models to evaluation mode
                for m in models:
                    m.eval()

                with jt.no_grad():
                    # forward pass (including logging)
                    __ = forward(
                        batch=batch, encoder=encoder, decoder=decoder, conf=conf,
                        is_valdt=True, step=valdt_step, epoch=epoch, batch_ind=valdt_batch_ind, num_batch=valdt_num_batch, start_time=start_time,
                        log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=valdt_writer,
                        lr=optimizer.param_groups[0].get('lr'), flog=flog)

    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write("Saving final checkpoint ...... ")
    util.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    print("DONE")
    flog.write("DONE\n")

    flog.close()

def forward(batch, encoder, decoder, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):
    objects = batch

    losses = {
        'latent': jt.zeros(1),
        'geo': jt.zeros(1),
        'center': jt.zeros(1),
        'scale': jt.zeros(1),
        'type': jt.zeros(1),
        'kldiv': jt.zeros(1)
    }

    # process every data in the batch individually
    for obj in objects:

        # encode object to get root code
        root_code = encoder.encode_structure(obj=obj)

        # get kldiv loss
        if not conf.non_variational:
            root_code, obj_kldiv_loss = jt.chunk(root_code, 2, 1)
            obj_kldiv_loss = -obj_kldiv_loss.sum() # negative kldiv, sum over feature dimensions
            losses['kldiv'] = losses['kldiv'] + obj_kldiv_loss
        # decode root code to get reconstruction loss
        obj_losses = decoder.structure_recon_loss(z=root_code, gt_tree=obj)
        # with jt.no_grad():
        #     recon_obj = decoder.decode_structure(z=root_code, model_name=obj.name)
        #     print(recon_obj)
        for loss_name, loss in obj_losses.items():
            losses[loss_name] = losses[loss_name] + loss

    for loss_name in losses.keys():
        losses[loss_name] = losses[loss_name] / len(objects)

    losses['latent'] *= conf.loss_weight_latent
    losses['geo'] *= conf.loss_weight_geo
    losses['center'] *= conf.loss_weight_center
    losses['scale'] *= conf.loss_weight_scale
    losses['type'] *= conf.loss_weight_type
    losses['kldiv'] *= conf.loss_weight_kldiv

    total_loss = 0
    for loss in losses.values():
        total_loss += loss

    with jt.no_grad():
        # log to console
        if log_console:
            print(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['latent'].item():>11.2f} '''
                f'''{losses['geo'].item():>11.2f} '''
                f'''{losses['center'].item():>11.2f} '''
                f'''{losses['scale'].item():>11.2f} '''
                f'''{losses['type'].item():>11.2f} ''' 
                f'''{losses['kldiv'].item():>10.2f} '''
                f'''{total_loss.item():>10.2f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['latent'].item():>11.2f} '''
                f'''{losses['geo'].item():>11.2f} '''
                f'''{losses['center'].item():>11.2f} '''
                f'''{losses['scale'].item():>11.2f} '''
                f'''{losses['type'].item():>11.2f} '''
                f'''{losses['kldiv'].item():>10.2f} '''
                f'''{total_loss.item():>10.2f}''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)
            tb_writer.add_scalar('latent_loss', losses['latent'].item(), step)
            # tb_writer.add_scalar('geo_loss', losses['geo'].item(), step)
            # tb_writer.add_scalar('center_loss', losses['center'].item(), step)
            # tb_writer.add_scalar('scale_loss', losses['scale'].item(), step)
            tb_writer.add_scalar('type_loss', losses['type'].item(), step)
            tb_writer.add_scalar('kldiv_loss', losses['kldiv'].item(), step)

    return total_loss

if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = add_train_vae_args(parser)
    config = parser.parse_args()

    train(config)

