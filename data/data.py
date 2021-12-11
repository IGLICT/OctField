import os
import json
import h5py
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from argparse import ArgumentParser
from utils.config import add_data_args
import utils.util as util

"""
dataset config
    preload
    occ
    data_path
    subsample
    voxel_res
    train/valdt_split
    
python data.py  --data_path /home/tangjiaheng/mnt/7/data/  \
        --train_list ../data_split/02691156_0.0_test.txt \
        --train_dataset Plane_test_4096_3_4096_0.0_16.h5 \
  --check_valid \
"""
DEBUG = False
###############################################################################
# Data structure
###############################################################################

class Tree():
    class Node():
        def __init__(self, parent=None):
            self.type = None
            self.global_id = 0
            self.geo_feat = None
            self.isleaf = True
            self.children = []
            self.parent = None
            self.voxels = None
            self.normals = None
            self.points = None
            self.values = None
            self.pred = None
            self.volume = None
            self.cell = None
            self.depth = -1
            self.pos = None
            self.device = 'cpu'
            self.connection = []
            

        def get_cell(self, local_idx):
            center = jt.zeros(3).to(self.device)
            scale = self.cell[3:6] / 2
            x = local_idx % 2
            y = (local_idx // 2) % 2
            z = local_idx // 4
            center[0] = self.cell[0] + (z - 0.5) * scale[0]
            center[1] = self.cell[1] + (y - 0.5) * scale[1]
            center[2] = self.cell[2] + (x - 0.5) * scale[2]
            return jt.concat([center, scale])

        def _to_str(self, level, pid):
            if self.type is None:
                out_str = ''
            else:
                type_str = ''
                for i in range(8):
                    type_str += str(int(self.type[i]))
                connection_str = ''
                for i in range(6):
                    connection_str += f' <{None if self.connection[i] is None else self.connection[i].global_id}>'
                out_str = f'''{'  |'*(level-1)}{'  ├'*(level > 0)}{str(pid)} <{self.global_id}>''' \
                            f'''{' [LEAF] ' if self.isleaf else type_str} {connection_str}\n'''
            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx)
            return out_str

        def __str__(self):
            return self._to_str(0, 0)


        def BuildConnection(self):
            for i, exists in enumerate(self.type):
                if (exists == 1):
                    self.children[i].BuildConnection()
            y_index = [0, 1, 4, 5]
            for i in range(4):
                if self.type[i * 2] and self.type[i * 2 + 1]:
                    self.ConnectTree(self.children[i * 2], self.children[i * 2 + 1], 2)
                if self.type[y_index[i]] and self.type[y_index[i] + 2]:
                    self.ConnectTree(self.children[y_index[i]], self.children[y_index[i] + 2], 1)
                if self.type[i] and self.type[i + 4]:
                    self.ConnectTree(self.children[i], self.children[i + 4], 0)
            return self

        def ConnectTree(self, l, r, dim):
            y_index = (0, 1, 4, 5)
            if dim == 2:
                l.connection[2] = r
                r.connection[5] = l
                for i in range(4):
                    if l.type[i * 2 + 1] and r.type[i * 2]:
                        self.ConnectTree(l.children[i * 2 + 1], r.children[i * 2], dim)
            elif dim == 1:
                l.connection[1] = r
                r.connection[4] = l
                for i in range(4):
                    if l.type[y_index[i] + 2] and r.type[y_index[i]]:
                        self.ConnectTree(l.children[y_index[i] + 2], r.children[y_index[i]], dim)
            elif dim == 0:
                l.connection[0] = r
                r.connection[3] = l
                for i in range(4):
                    if l.type[i + 4] and r.type[i]:
                        self.ConnectTree(l.children[i + 4], r.children[i], dim)
            return self


    def __init__(self, max_depth, overlap=0.0, name='unnamed'):
        self.max_depth = max_depth
        self.overlap = overlap
        self.name = name
        self.root = Tree.Node()
        self.volume = None
        self.mask = None
        self.latent = None
        self.device = 'cpu'

    def get_root_cell(self):
        return self.root.cell

    def build_tree(self, children, cells, voxels=None, normals=None, points=None, values=None):
        queue = [(self.root, 0, 0, 0, 0, 0)]
        dx = (0,0,0,0,1,1,1,1)
        dy = (0,0,1,1,0,0,1,1)
        dz = (0,1,0,1,0,1,0,1)
        id_cnt = 0
        while queue:
            node, cur, depth, x, y, z = queue.pop()
            node.global_id = id_cnt
            id_cnt += 1
            node.cell = cells[cur]
            if voxels is not None:
                node.voxels = voxels[cur]
            if normals is not None:
                node.normals = normals[cur]
            if points is not None:
                node.points = points[cur]
            if values is not None:
                node.values = values[cur]
            node.type = (children[cur] > 0).float()
            node.depth = depth
            node.pos = (x, y, z)
            node.children = [Tree.Node(node) for i in range(8)]
            node.connection = [None] * 6
            if depth < self.max_depth:
                for i, c in enumerate(children[cur]):
                    node.children[i].parent = node
                    if c > 0:
                        queue.append((
                            node.children[i], c, depth + 1,
                            x * 2 + dx[i], y * 2 + dy[i], z * 2 + dz[i]
                        ))
            else:
                node.type[:] = 0
            if node.type.sum() > 0:
                node.isleaf = False
        self.root.BuildConnection()
        return self

    def __str__(self):
        return self.root.__str__()

    def get_leaves(self, max_depth=3):
        queue = [self.root]
        leaves = []
        while queue:
            node = queue.pop()
            if node.isleaf or (node.depth == max_depth):
                leaves.append(node)
            else:
                for i, exists in enumerate(node.type):
                    if (exists == 1):
                        queue.append(node.children[i])
        return leaves


    def output_obj(self, output_dir='.', mode='gt', level=0.5):
        tar_file = os.path.join(output_dir, f'{self.name}.obj')
        leaves = self.get_leaves()
        points = []
        values = []
        for leaf in leaves:
            center = leaf.cell[0:3]
            scale = leaf.cell[3:6]
            p = (leaf.points * scale / 2+ center).cpu().numpy()
            if mode == 'gt':
                v = -leaf.values.cpu().numpy() + 1
            elif mode == 'pred':
                v = -leaf.pred.cpu().numpy() + 1
            points.append(p)
            values.append(v)
        points = np.concatenate(points)
        values = np.concatenate(values)
        meshExtractor = util.DelauneyMeshExtractor(points, values, threshold=level)
        v, f = meshExtractor.extract_mesh()
        if f.shape[0] > 0:
            print(f'model {self.name} ==> file {tar_file}')
            util.export_obj(tar_file, v, f+1)
            return True
        else:
            print(f'model {self.name} (no face)')
            return False


###############################################################################
# Dataset
###############################################################################


class ShapeDataset(Dataset):
    def __init__(self, conf, split='train', batch_size=1, shuffle=False):
        super(ShapeDataset, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_depth = conf.max_depth
        self.overlap = conf.overlap
        self.data_path = conf.data_path
        if split == 'train':
            self.is_train = True
            item_list = conf.train_list
            self.data_h5 = conf.train_dataset
        elif split == 'val':
            self.is_train = False
            item_list = conf.val_list
            self.data_h5 = conf.val_dataset
        elif split == 'test':
            self.is_train = False
            item_list = conf.test_list
            self.data_h5 = conf.test_dataset
        with open(item_list, 'r') as f:
            self.items_lst = [item.rstrip() for item in f]
        self.keys = ['children', 'cells', 'points', 'values', 'normals', 'voxels']
        self.idx_end, self.data_dict = self.load_h5f(conf.load_ram)

    def load_h5f(self, load_ram=True):
        src_file = os.path.join(self.data_path, self.data_h5)
        h5f = h5py.File(src_file, 'r')
        idx_end = [0] + list(h5f.get('idx_end'))
        return idx_end, {
            key: np.array(h5f.get(key)) for key in self.keys
        }

    def __getitem__(self, index):
        model_name = self.items_lst[index]
        idx = slice(self.idx_end[index], self.idx_end[index + 1])
        children = self.data_dict['children'][idx]
        cells = self.data_dict['cells'][idx]
        normals = self.data_dict['normals'][idx]
        voxels = self.data_dict['voxels'][idx]
        points = self.data_dict['points'][idx]
        values = self.data_dict['values'][idx]
        num_cells, num_points, _ = points.shape
        half = num_points // 2
        for i in range(num_cells):
            pos = np.nonzero(values[i]>0.5)[0]
            neg = np.nonzero(values[i]<0.5)[0]
            resample = np.zeros(num_points, dtype=np.long)
            resample[:half] = pos[np.random.randint(pos.shape[0],size=(half,))].reshape(-1)
            resample[half:] = neg[np.random.randint(neg.shape[0],size=(half,))].reshape(-1)
            points[i] = points[i, resample]
            values[i] = values[i, resample]
        tree = Tree(self.max_depth, self.overlap, model_name)
        tree.build_tree(children, cells, voxels, normals, points, values)
        return tree

    def __len__(self):
        return len(self.items_lst)

    def collate_batch(self, batch):
        return batch
