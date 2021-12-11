
import os
import time
import sys
import shutil
import random
import h5py
import logging
from time import strftime
from argparse import ArgumentParser
import numpy as np
import utils
from utils import config
"""
python pkg_part.py --category 'Plane_new_train' \
    --num_smp 8192 \
    --depth 3 \
    --res 16384 \
    --overlap 0.5 \
    --vres 16 \
    --data_path /mnt/9/tangjiaheng/data/shapenet/ShapeNetCore.v2/02691156/  \
    --train_dataset ../data_split/02691156_train.txt \
    --store_dir /mnt/7/tangjiaheng/data/

python pkg_part.py --category 'Plane_new_val' \
    --data_path /mnt/9/tangjiaheng/data/shapenet/ShapeNetCore.v2/02691156/  \
    --num_smp 8192 \
    --depth 3 \
    --res 16384 \
    --overlap 0.5 \
    --vres 16 \
    --train_dataset ../data_split/02691156_val.txt \
    --store_dir /mnt/7/tangjiaheng/data/

python pkg_part.py --category 'Plane_new_test' \
    --data_path /mnt/9/tangjiaheng/data/shapenet/ShapeNetCore.v2/02691156/  \
    --num_smp 8192 \
    --depth 3 \
    --res 16384 \
    --overlap 0.5 \
    --vres 16 \
    --train_dataset ../data_split/02691156_test.txt \
    --store_dir /mnt/7/tangjiaheng/data/
"""



def get_npz_info(conf, model_path):
    npzfile = os.path.join(
        model_path, 'models',
        f'samples_{conf.num_smp}_{conf.depth}_{conf.res}_{conf.overlap}.npz'
    )
    voxels_npy = os.path.join(
        model_path, 'models',
        f'voxels_{conf.num_smp}_{conf.depth}_{conf.res}_{conf.overlap}_{conf.vres}.npy'
    )
    avgnormals_npy = os.path.join(
        model_path, 'models',
        f'normals_{conf.num_smp}_{conf.depth}_{conf.res}_{conf.overlap}_{conf.vres}.npy'
    )
    arr = np.load(npzfile)
    normals = np.load(avgnormals_npy).astype(np.float32)
    normals[np.isnan(normals)]=0
    normals[np.isinf(normals)]=0
    
    nodes = arr['octree_node']
    cells = arr['occupied_cells'].astype(np.float32)
    sample_points = arr['samples']
    num_cells = cells.shape[0]
    num_smp = conf.num_smp
    
    children = -np.ones([num_cells, 8], dtype=np.int16)
    node2cell = {0: 0}
    for i, node in enumerate(nodes):
        parent, _, local_id, idx = node
        node2cell[i] = idx
        if i > 0:
            children[node2cell[parent], local_id] = idx
    points = np.zeros([num_cells, num_smp, 3], dtype=np.float32)
    values = np.zeros([num_cells, num_smp], dtype=np.float32)
    for idx in range(num_cells):
        scale = cells[idx, 3:6] * (conf.overlap * 2 + 1)
        center = cells[idx, 0:3]
        points[idx] = (sample_points[idx, :, 0:3] - center) * 2 / scale # -1,1
        values[idx] = sample_points[idx, :, 3]
    data_dict = {
        'children': children,
        'cells': cells,
        'points': points,
        'values': values,
        'normals': normals,
        'voxels': voxels
    }
    return num_cells, data_dict


def create_h5(conf, split):
    with open(split, 'r') as fd:
        items_lst = [item.rstrip() for item in fd]
    data_h5 = f'{conf.category}_{conf.num_smp}_{conf.depth}_{conf.res}_{conf.overlap}_{conf.vres}.h5'
    tar_file = os.path.join(conf.store_dir, data_h5)
    VR = conf.vres
    num_smp = conf.num_smp
    num_items = len(items_lst)
    h5f = h5py.File(tar_file, 'a')
    h5f.create_dataset('idx_end', (num_items,), dtype=np.int64, compression='gzip')
    h5f.create_dataset('children', (0, 8), dtype=np.int16, maxshape=(None, 8), compression='gzip')
    h5f.create_dataset('cells', (0, 6), dtype=np.float32, maxshape=(None, 6), compression='gzip')
    h5f.create_dataset('points', (0, num_smp, 3), dtype=np.float32, maxshape=(None, num_smp, 3), compression='gzip')
    h5f.create_dataset('values', (0, num_smp), dtype=np.float32, maxshape=(None, num_smp), compression='gzip')
    h5f.create_dataset('normals', (0, 3), dtype=np.float32, maxshape=(None, 3), compression='gzip')
    h5f.create_dataset('voxels', (0, VR, VR, VR), dtype=np.bool, maxshape=(None, VR, VR, VR), compression='gzip')
    cnt = 0
    for i, item in enumerate(items_lst):
        model_path = os.path.join(conf.data_path, item)
        num_cells, data_dict = get_npz_info(conf, model_path)
        cnt += num_cells
        h5f['idx_end'][i] = cnt
        for key in data_dict.keys():
            h5f[key].resize(cnt, axis=0)
            h5f[key][-num_cells:] = data_dict[key]
        print(f'{item} done')
    print('Finished')
    print(num_items)
    for key in data_dict.keys():
        print(f'Shape of {key}: {h5f[key].shape}')
        h5f[key][-num_cells:] = data_dict[key]
    h5f.close()
    return num_items, cnt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = config.add_base_args(parser)
    parser = config.add_data_args(parser)
    parser.add_argument('--store_dir', type=str, required=True)
    parser.add_argument('--train_dataset', type=str, required=True)
    config = parser.parse_args()
    _ = create_h5(conf=config, split=config.train_dataset)
