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
    avgnormals_npy = os.path.join(
        model_path, 'models',
        f'normals_{num_smp}_{depth}_{res}_{overlap}.npy'
    )
    if (os.path.exists(avgnormals_npy)) and (os.path.getsize(avgnormals_npy) > 0):
        print(f'{model_path} done')
    else:
        
        arr = np.load(npzfile)
        cells = arr['occupied_cells']
        mesh = trimesh.load(manifold_obj, force='mesh')
        mesh = mesh.process()
        v = mesh.vertices
        vn = mesh.vertex_normals
        num_cells = cells.shape[0]
        avgnormals = np.zeros([num_cells, 3])
        for i in range(num_cells):
            center = cells[i, 0:3]
            scale = cells[i, 3:6]
            min_corner = center - scale * (0.5 + overlap)
            max_corner = center + scale * (0.5 + overlap)
            bounding_box = [min_corner, max_corner]
            inside = trimesh.bounds.contains(bounding_box, v)
            inside = np.array(inside).reshape(-1,1)
            vn_avg = np.sum(vn * inside, axis=0)
            vn_avg_len = np.linalg.norm(vn_avg)
            if vn_avg_len > 0.01:
                vn_avg /= vn_avg_len
            else:
                vn_avg[:] = 0
            avgnormals[i] = vn_avg
        np.save(avgnormals_npy, avgnormals)
        print(f'{model_path} normal done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config.add_check_args(parser)
    conf = parser.parse_args()
    num_smp=conf.num_smp
    depth=conf.depth
    res=conf.res
    overlap=conf.overlap
    with open(conf.success_list, 'r') as fd:
        items_lst = [item.rstrip() for item in fd]
    success_lst = [
        os.path.join(conf.data_source, item)
        for item in items_lst
    ]
    with poolcontext(processes=conf.num_workers) as pool:
        pool.map(voxelize_cells, success_lst)
        
"""
python normal_tool.py --num_smp 4096 \
    --depth 3 \
    --res 4096 \
    --overlap 0.5 \
    --data_source /mnt/8/tangjiaheng/data/shapenet/ShapeNetCore.v2/02958343 \
    --success_list data_split/02958343_valid.txt \
    --num_workers 16
"""