import os
import argparse
import multiprocessing
from multiprocessing import Pool
from contextlib import contextmanager
import trimesh
import numpy as np

from utils import config
from utils.util import genSamples


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def voxelize_cells(model_path):
    npzfile = os.path.join(
        model_path, 'models',
        f'samples_{num_smp}_{depth}_{res}_{overlap}.npz'
    )
    manifold_obj = os.path.join(model_path, 'models/manifold.obj')
    voxels_npy = os.path.join(
        model_path, 'models',
        f'voxels_{num_smp}_{depth}_{res}_{overlap}_{vres}.npy'
    )
    if (os.path.exists(voxels_npy)) and (os.path.getsize(voxels_npy) > 0):
        print(f'{model_path} done')
    else:
        try:
            arr = np.load(npzfile)
            cells = arr['occupied_cells']
            mesh = trimesh.load(manifold_obj, force='mesh')
            mesh = mesh.process()
            num_cells = cells.shape[0]
            voxels = np.zeros([num_cells, vres, vres, vres])
            for i in range(num_cells):
                center = cells[i, 0:3]
                scale = cells[i, 3:6]
                min_corner = center - scale * (0.5 + overlap)
                max_corner = center + scale * (0.5 + overlap)
                bounding_box = np.concatenate([min_corner, max_corner]) 
                voxel = mesh.voxelized(
                    method='binvox', pitch=-1, 
                    exact=True, dimension=vres, center=True,
                    bounding_box=bounding_box)
                voxels[i] = voxel.matrix
            np.save(voxels_npy, voxels)
            print(f'{model_path} done')
        except:
            print(f'{model_path} missed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config.add_voxelize_args(parser)
    conf = parser.parse_args()
    num_smp=conf.num_smp
    depth=conf.depth
    res=conf.res
    overlap=conf.overlap
    vres=conf.vres
    with open(conf.success_list, 'r') as fd:
        items_lst = [item.rstrip() for item in fd]
    success_lst = [
        os.path.join(conf.data_source, item)
        for item in items_lst
    ]
    with poolcontext(processes=conf.num_workers) as pool:
        pool.map(voxelize_cells, success_lst)

"""
python voxelize_tool.py --num_smp 4096 \
    --depth 3 \
    --res 4096 \
    --vres 16 \
    --overlap 0.5 \
    --data_source /home/tangjiaheng/mnt/8/data/shapenet/ShapeNetCore.v2/02958343 \
    --success_list data_split/02958343.txt
"""

