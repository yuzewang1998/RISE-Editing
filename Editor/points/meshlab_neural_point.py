from Editor.points.base_point import BasePoint
import os
from plyfile import PlyData, PlyElement
import numpy as np
from copy import deepcopy
from tqdm import tqdm
class MeshlabNeuralPoint(BasePoint):
    def __init__(self,opt):
        super().__init__(opt)

    def load_from_ply(self, name):
        self.ply_path = os.path.join(self.file_dir, name + '_meshlabpoint.ply')
        assert os.path.exists(self.ply_path), '{}_meshlabpoint doesn`t exist ,check!'.format(name)
        print('loading neural point cloud from ply....')
        plydata = PlyData.read(self.ply_path)
        x, y, z = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
            plydata.elements[0].data["y"].astype(np.float32)), np.array(
            plydata.elements[0].data["z"].astype(np.float32))
        self.xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)

    def save_as_ply(self,name):
        assert self.xyz is not None, '[ERROR]Save before load,check it!'
        print('Saving neural point cloud as ply...', self.xyz.shape)
        vertex = np.concatenate([self.xyz],
                                axis=-1)
        vertex = [tuple(i) for i in vertex]
        # ply的格式，没写循环、增加可读性
        vertex = np.array(vertex,
                          dtype=[
                              ("x", np.dtype("float32")),
                              ("y", np.dtype("float32")),
                              ("z", np.dtype("float32")),
                          ]
                          )
        ply_pc = PlyElement.describe(vertex, "vertex")
        ply_pc = PlyData([ply_pc])
        self.save_ply_path = os.path.join(self.file_dir, name + '_meshlabpoint.ply')
        ply_pc.write(self.save_ply_path)
        print('Save done')

    def down_sample(self,voxel_size = np.array([0.008,0.008,0.008]),max_ptr_in_a_occ = 16,down_sample_func='average'):
        lower = np.min(self.xyz, axis=0)
        upper = np.max(self.xyz, axis=0)
        bbox = np.concatenate([lower[..., None], upper[..., None]], axis=-1)
        occ_shift = bbox[:,0]-1e-9
        xyz = deepcopy(self.xyz)

        shifted_neural_ptr = xyz - occ_shift
        shifted_neural_ptr_occ_coord = np.floor(shifted_neural_ptr / voxel_size).astype(int)  # [n,3] int
        occ_shape = np.ceil((bbox[:, 1] - bbox[:, 0]) / voxel_size).astype(int)
        occ_index = np.zeros([occ_shape[0], occ_shape[1], occ_shape[2], max_ptr_in_a_occ],
                             dtype=np.int32)  # e.g[60,50,40,32]
        occ_mask = np.zeros(occ_shape, dtype=np.int32)
        occ_mask_bool = np.zeros(occ_index.shape,dtype=np.bool)
        # build occ
        for i in range(len(shifted_neural_ptr)):
            coord_x, coord_y, coord_z = shifted_neural_ptr_occ_coord[i]  # [3,]
            occ_mask_bool[coord_x, coord_y, coord_z, occ_mask[coord_x, coord_y, coord_z]] = True
            occ_index[coord_x, coord_y, coord_z, occ_mask[coord_x, coord_y, coord_z]] = i
            occ_mask[coord_x, coord_y, coord_z] += 1

        # quick search
        down_sampled_xyz  = np.zeros([0, 3],dtype=np.float32)

        xyz_reshape = xyz[occ_index]
        if down_sample_func == 'random':
            for h in tqdm(range(occ_mask.shape[0])):
                for w in range(occ_mask.shape[1]):
                    for c in range(occ_mask.shape[2]):
                        index = occ_mask[h][w][c]
                        if index!=0:
                            tmp_xyz = xyz_reshape[h][w][c][index-1]
                            down_sampled_xyz = np.concatenate([down_sampled_xyz,tmp_xyz[None,...]],axis=0)
        if down_sample_func == 'average':
            print(xyz_reshape.shape)
            xyz_reshape_cumsum = np.cumsum( xyz_reshape,axis = 3)
            for h in tqdm(range(occ_mask.shape[0])):
                for w in range(occ_mask.shape[1]):
                    for c in range(occ_mask.shape[2]):
                        index = occ_mask[h][w][c]
                        if index!=0:
                            tmp_xyz = xyz_reshape_cumsum[h][w][c][index-1]/(index)
                            down_sampled_xyz = np.concatenate([down_sampled_xyz,tmp_xyz[None,...]],axis=0)
        new_ptr = deepcopy(self)
        new_ptr.xyz = down_sampled_xyz
        print(down_sampled_xyz)
        return new_ptr
