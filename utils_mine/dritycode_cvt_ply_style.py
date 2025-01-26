from Editor.points.base_neural_point import BaseNeuralPoint
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from Editor.points.meshlab_neural_point import MeshlabNeuralPoint
from Editor.points import create_neural_point,find_neural_point_id
from copy import deepcopy

scannet_ply_path = '/home/slam/devdata/npbgpp/npbgplusplus/data/scannet/scan241/scannet_style.ply'
sv_path = '/home/slam/devdata/npbgpp/npbgplusplus/data/scannet/scan241/full.ply'
plydata = PlyData.read(scannet_ply_path)
x, y, z = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
    plydata.elements[0].data["y"].astype(np.float32)), np.array(
    plydata.elements[0].data["z"].astype(np.float32))
xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)
r, g, b = np.array(plydata.elements[0].data["red"].astype(np.float32)), np.array(
    plydata.elements[0].data["green"].astype(np.float32)), np.array(
    plydata.elements[0].data["blue"].astype(np.float32))
color = np.concatenate([r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]], axis=-1)
vertex = np.concatenate([xyz, color], axis=-1)
vertex = [tuple(i) for i in vertex]
vertex = np.array(vertex,
                  dtype=[
                      ("x", np.dtype("float32")),
                      ("y", np.dtype("float32")),
                      ("z", np.dtype("float32")),
                      ("red", np.dtype("float32")),
                      ("green", np.dtype("float32")),
                      ("blue", np.dtype("float32")),
                  ]
                  )
ply_pc = PlyElement.describe(vertex, "vertex")
ply_pc = PlyData([ply_pc])
ply_pc.write(sv_path)
print('Save done')