
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
import config


def get_npz_info(conf, model_name):
    npzfile = os.path.join(
        # conf.data_path,
        # f'{model_name}_{conf.num_smp}_{conf.depth}_{conf.res}{"_0" if conf.overlap==0 else ""}.npz'
        f'{model_name}.npz'
    )
    arr = np.load(npzfile)
    children = arr['children'].astype(np.int16)
    levels = arr['levels'].astype(np.int8).reshape(-1,1)
    cells = arr['cells'].astype(np.float32)
    samples = arr['samples'].astype(np.float32)
    voxels = arr['voxels'].astype(np.bool)
    normals = arr['normals'].astype(np.float32)
    num_cells = cells.shape[0]
    num_smp = conf.num_smp

    points = np.zeros([num_cells, num_smp, 3], dtype=np.float32)
    values = np.zeros([num_cells, num_smp], dtype=np.float32)
    for idx in range(num_cells):
        scale = cells[idx, 3:6] * (conf.overlap * 2 + 1)
        center = cells[idx, 0:3]
        points[idx] = (samples[idx, :, 0:3] - center) * 2 / scale # -1,1
        values[idx] = samples[idx, :, 3]
    data_dict = {
        'children': children,
        'levels': levels,
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
    for item in items_lst:
        data_h5 = f'{item}.h5'
        tar_file = os.path.join(conf.store_dir, data_h5)
        VR = conf.vres
        num_smp = conf.num_smp
        # num_items = len(items_lst)
        num_items = 1
        h5f = h5py.File(tar_file, 'a')
        h5f.create_dataset('idx_end', (num_items,), dtype=np.int64, compression='gzip')
        h5f.create_dataset('children', (0, 8), dtype=np.int16, maxshape=(None, 8), compression='gzip')
        h5f.create_dataset('levels', (0, 1), dtype=np.int8, maxshape=(None, 1), compression='gzip')
        h5f.create_dataset('cells', (0, 6), dtype=np.float32, maxshape=(None, 6), compression='gzip')
        h5f.create_dataset('points', (0, num_smp, 3), dtype=np.float32, maxshape=(None, num_smp, 3), compression='gzip')
        h5f.create_dataset('values', (0, num_smp), dtype=np.float32, maxshape=(None, num_smp), compression='gzip')
        h5f.create_dataset('normals', (0, 3), dtype=np.float32, maxshape=(None, 3), compression='gzip')
        h5f.create_dataset('voxels', (0, VR, VR, VR), dtype=np.bool, maxshape=(None, VR, VR, VR), compression='gzip')
        cnt = 0
        # for i, item in enumerate(items_lst):
        model_path = os.path.join(conf.data_path, item)
        num_cells, data_dict = get_npz_info(conf, model_path)
        cnt += num_cells
        h5f['idx_end'][0] = cnt
        for key in data_dict.keys():
            h5f[key].resize(cnt, axis=0)
            h5f[key][-num_cells:] = data_dict[key]
        print(f'{item} done')
        print('Finished')
        print(num_items)
        for key in data_dict.keys():
            print(f'Shape of {key}: {h5f[key].shape}')
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
