import os
import numpy as np
import importlib
from scipy.spatial import Delaunay
from itertools import combinations
import jittor as jt
import skimage
from skimage import measure
import trimesh
from trimesh.voxel import creation
import imp_sampling
import sys

def save_checkpoint(
    models, model_names,
    dirname, epoch=None, prepend_epoch=False,
    optimizers=None, optimizer_names=None
):
    if len(models) != len(model_names) or (
        (optimizers is not None) and (
            len(optimizers) != len(optimizer_names))
    ):
        raise ValueError(
            'Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        jt.save(model.state_dict(), os.path.join(dirname, filename))

    if optimizers is not None:
        filename = 'checkpt.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        checkpt = {'epoch': epoch}
        for opt, optimizer_name in zip(optimizers, optimizer_names):
            checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
        jt.save(checkpt, os.path.join(dirname, filename))
        

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        model.load_state_dict(jt.load(os.path.join(dirname, filename)), strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = os.path.join(dirname, 'checkpt.pth')
        if epoch is not None:
            filename = f'{epoch}_' + filename
        if os.path.exists(filename):
            checkpt = jt.load(filename)
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch


def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module(model_version)


def create_cube(center, scale):
    cube_verts = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]]
    cube_tri = [
        [2, 1, 0], [1, 2, 3],
        [4, 2, 0], [2, 4, 6],
        [1, 4, 0], [4, 1, 5],
        [6, 5, 7], [5, 6, 4],
        [3, 6, 7], [6, 3, 2],
        [5, 3, 7], [3, 5, 1]]
    verts = np.array(cube_verts)
    verts = verts * scale + center
    faces = np.array(cube_tri)
    return verts, faces


def voxel_to_world(voxel, center, scale):
    res = voxel.shape[0]
    x, y, z = np.nonzero(voxel)
    sparse_voxel = np.vstack([x, y, z]).transpose()

    sparse_voxel_world = ((sparse_voxel.astype(float) + 0.5) / res - 0.5) * scale + center
    return sparse_voxel_world


def voxel_to_mesh(voxel, center, scale):
    res = voxel.shape[0]
    sparse_voxel = voxel_to_world(voxel, center, scale)
    verts = []
    faces = []
    for cnt, box in enumerate(sparse_voxel):
        v, f = create_cube(box, scale/res)
        verts.append(v)
        faces.append(f + cnt * 8)

    verts = np.vstack(verts)
    faces = np.vstack(faces)
    return verts, faces


def mesh_to_world(verts, faces, center, scale, mc_res=0, overlap=0.5):
    if mc_res > 1:
        verts = (verts / (mc_res - 1) - 0.5) * scale + center
    else:
        verts = verts * scale + center
    return verts, faces


def marching_cubes_to_mesh(values, center, scale, level=0, world=True):
    mc_res = values.shape[0]
    verts, faces, _, _ = skimage.measure.marching_cubes(volume=values, level=level)
    if world:
        return mesh_to_world(verts, faces, center, scale, mc_res)
    else:
        return verts, faces



def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts


def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))


def export_pts_rgb(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], v[i, 3], v[i, 4], v[i, 5]))


def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f


def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))


def export_obj_quad(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2], f[i, 3]))


def load_sdf(fn, split=True):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        tot = int(lines[0])
        assert(tot == len(lines)-1)
        samples = []
        for line in lines[1:]:
            samples.append(np.float32(line.split()[0:5]))
        samples = np.vstack(samples)

    if split:
        pts, occ, sdf = np.split(samples, [3, 4], axis=1)
        return pts, occ, sdf
    else:
        return samples


def load_sdf_new(fn, split=True):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        num_cells, tot = lines[0].split()
        num_cells = int(num_cells)
        tot = int(tot)
        cells = []
        for line in lines[1:num_cells+1]:
            cells.append(np.float32(line.split()[0:6]))
        samples = []
        for line in lines[num_cells+1:]:
            samples.append(np.float32(line.split()[0:6]))
        cells = np.vstack(cells)
        samples = np.vstack(samples)

    if split:
        corner, scale = np.split(cells, [3], axis=1)
        pts, occ, sdf, cell = np.split(samples, [3, 4, 5], axis=1)
        return corner, scale, pts, occ, sdf, cell
    else:
        return cells, samples


def load_sdf_tree(fn, split=True):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        num_nodes, num_cells, tot = lines[0].split()
        num_nodes = int(num_nodes)
        num_cells = int(num_cells)
        tot = int(tot)
        nodes = []
        for line in lines[1:num_nodes + 1]:
            nodes.append(np.int32(line.split()[0:4]))
        cells = []
        for line in lines[num_nodes + 1:num_nodes + num_cells + 1]:
            cells.append(np.float32(line.split()[0:6]))
        samples = []
        for line in lines[num_nodes + num_cells + 1:]:
            samples.append(np.float32(line.split()[0:6]))
        nodes = np.vstack(nodes)
        cells = np.vstack(cells)
        samples = np.vstack(samples)

    if split:
        corner, scale = np.split(cells, [3], axis=1)
        pts, occ, sdf = np.split(samples, [3, 4, 5], axis=1)
        return corner, scale, pts, occ, sdf, nodes
    else:
        return cells, samples, nodes


def sdf_to_pts(pts, sdf, level=0.):
    pos, _ = np.nonzero(sdf > level)
    print(pos.shape)
    neg, _ = np.nonzero(sdf <= level)
    print(neg.shape)
    return pts[pos], pts[neg]


def genSamples(manifold_obj, num_smp=2048, depth=4, res=1024, overlap=0.):
    sampler = imp_sampling.SamplesGenerator(
        manifold_obj, num_smp, depth, res, overlap)
    cells = sampler.getTreeCells()
    nodes = sampler.getTreeNodes()
    sample_points = sampler.getSamplePoints()
    cell_num = cells.shape[0]
    sample_points = sample_points.reshape(cell_num, num_smp, 5)
    return nodes, cells, sample_points
