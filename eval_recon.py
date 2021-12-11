import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import jittor as jt
import utils
import utils.util as util
from utils.config import add_eval_args
from data.data import ShapeDataset, Tree

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()

# load train config
conf = jt.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path


# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

# load model
models = util.get_model_module(conf.model_version)

# set up device
jt.use_cuda = True  
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
if os.path.exists(os.path.join(conf.result_path, conf.exp_name)):
    response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
    if response != 'y':
        sys.exit()
    shutil.rmtree(os.path.join(conf.result_path, conf.exp_name))

# create a new directory to store eval results
os.makedirs(os.path.join(conf.result_path, conf.exp_name))

# create models
encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=False)
decoder = models.RecursiveDecoder(conf)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

# load pretrained model
__ = util.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

# create dataset
dataloader = ShapeDataset(conf, conf.test_dataset, batch_size=1, shuffle=False)

# set models to evaluation mode
for m in models:
    m.eval()

# test over all test shapes
num_batch = len(dataloader)
with jt.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        for obj in batch:
            root_code_and_kld = encoder.encode_structure(obj=obj)
            root_code, _ = jt.chunk(root_code_and_kld, 2, 1)
            print(root_code.shape)
            recon_obj = decoder.decode_structure(z=root_code_and_kld, model_name=obj.name)
#             recon_obj.output_obj(level=0.5, output_dir='recon')
            print('[%d/%d] ' % (batch_ind, num_batch), recon_obj.name)
