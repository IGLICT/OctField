
import os
import argparse
import multiprocessing
from multiprocessing import Pool
from contextlib import contextmanager

import numpy as np

from utils import config
from utils.util import genSamples


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def sample_check_save(model_name):
    model_path = os.path.join(conf.data_source, model_name)
    out_path = os.path.join(conf.output_dir, model_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    npzfile = os.path.join(
        out_path,
        f'samples_{num_smp}_{depth}_{res}_{overlap}.npz'
    )
    fail_flag = os.path.join(
        out_path,
        f'flag_{num_smp}_{depth}_{res}_{overlap}.txt'
    )

    if (os.path.exists(npzfile)) and (os.path.getsize(npzfile) > 0):
        print(f'{model_path} True')
        return True
    elif os.path.exists(fail_flag):
        print(f'{model_path} False')
        return False
    else:
        try:
            raw_obj = os.path.join(model_path, 'models/model_normalized.obj')
            manifold_obj = os.path.join(model_path, 'models/manifold.obj')
            if (not os.path.exists(manifold_obj))  or (os.path.getsize(manifold_obj) == 0):
                cmd = f'./manifoldplus --input {raw_obj} --output {manifold_obj}'
                # os.system(f'{cmd} > tmp')
                os.system(cmd)
            if (not os.path.exists(manifold_obj)):
                print(f'{model_path} Error')
                return False
            nodes, cells, sample_points = genSamples(
                manifold_obj,
                num_smp=num_smp,
                depth=depth,
                res=res,
                overlap=overlap
            )

        except:
            print(f'{model_path} Error')
            with open(fail_flag, 'w') as ff:
                ff.write(f'{model_path} False\n')
            try:
                del nodes
                del cells
                del sample_points
            except:
                pass
            return False

        else:
            inside = sample_points[:, :, 3]
            well_done = (inside.max(1).min() > 0.5) and (inside.min(1).max() < 0.5)
            del inside
            print(f'{model_path} {well_done}')
            if well_done:
                np.savez_compressed(
                    npzfile,
                    octree_node=nodes,
                    occupied_cells=cells,
                    samples=sample_points
                )
            else:
                with open(fail_flag, 'w') as ff:
                    ff.write(f'{model_path} False\n')
            del nodes
            del cells
            del sample_points
            return well_done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config.add_sample_args(parser)
    conf = parser.parse_args()
    num_smp=conf.num_smp
    depth=conf.depth
    res=conf.res
    overlap=conf.overlap
    with open(conf.data_list, 'r') as fd:
        items_lst = [item.rstrip() for item in fd]
    with poolcontext(processes=conf.num_workers) as pool:
        success = pool.map(sample_check_save, items_lst)
    with open(conf.success_list, 'w') as fs:
        for i, f in enumerate(items_lst):
            if success[i]:
                fs.write('%s\n' % (f))
    if conf.fail_list is not None:
        with open(conf.fail_list, 'w') as fs:
            for i, f in enumerate(items_lst):
                if not success[i]:
                    fs.write('%s\n' % (f))