from Editor.points.base_neural_point import BaseNeuralPoint
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from Editor.points.meshlab_neural_point import MeshlabNeuralPoint
from Editor.points import create_neural_point,find_neural_point_id
from copy import deepcopy
class PenerfNeuralPoint(BaseNeuralPoint):
    def __init__(self,opt):
        super().__init__(opt)
        self.dirx = None
        self.diry = None
        self.dirz = None

    def set_input(self,points_xyz,**kwargs):
        super().set_input(points_xyz,**kwargs)
        self.dirx = kwargs['points_dirx']
        self.diry = kwargs['points_diry']
        self.dirz = kwargs['points_dirz']

    def load_from_ply(self, name):
        self.load_ply_path = os.path.join(self.file_dir, name + '_penerfneuralpoint.ply')
        print(self.load_ply_path)
        assert os.path.exists(self.load_ply_path), '{}_penerfneuralpoint doesn`t exist ,check!'.format(name)
        print('loading neural point cloud from ply....')
        plydata = PlyData.read(self.load_ply_path)
        x, y, z = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
            plydata.elements[0].data["y"].astype(np.float32)), np.array(
            plydata.elements[0].data["z"].astype(np.float32))
        self.xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)
        r, g, b = np.array(plydata.elements[0].data["red"].astype(np.float32)), np.array(
            plydata.elements[0].data["green"].astype(np.float32)), np.array(
            plydata.elements[0].data["blue"].astype(np.float32))
        self.color = np.concatenate([r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]], axis=-1)
        dirx0, dirx1, dirx2 = np.array(plydata.elements[0].data["dirx0"].astype(np.float32)), np.array(
            plydata.elements[0].data["dirx1"].astype(np.float32)), np.array(
            plydata.elements[0].data["dirx2"].astype(np.float32))
        self.dirx = np.concatenate([dirx0[..., np.newaxis], dirx1[..., np.newaxis], dirx2[..., np.newaxis]], axis=-1)
        diry0, diry1, diry2 = np.array(plydata.elements[0].data["diry0"].astype(np.float32)), np.array(
            plydata.elements[0].data["diry1"].astype(np.float32)), np.array(
            plydata.elements[0].data["diry2"].astype(np.float32))
        self.diry = np.concatenate([diry0[..., np.newaxis], diry1[..., np.newaxis], diry2[..., np.newaxis]], axis=-1)
        dirz0, dirz1, dirz2 = np.array(plydata.elements[0].data["dirz0"].astype(np.float32)), np.array(
            plydata.elements[0].data["dirz1"].astype(np.float32)), np.array(
            plydata.elements[0].data["dirz2"].astype(np.float32))
        self.dirz = np.concatenate([dirz0[..., np.newaxis], dirz1[..., np.newaxis], dirz2[..., np.newaxis]], axis=-1)
        self.conf = np.array(plydata.elements[0].data["conf"].astype(np.float32))[...,None]
        self.embeding = np.zeros([self.xyz.shape[0],0])
        for i in range(32):
            tmp = np.array(plydata.elements[0].data["embeding"+str(i)].astype(np.float32))[...,np.newaxis]
            self.embeding =np.concatenate([self.embeding,tmp],axis = -1)
        print('loading done. Scale of neural point cloud:',self.embeding.shape[0])

    def save_as_ply(self,name):
        assert self.xyz is not None, '[ERROR]Save before load,check it!'
        print('Saving neural point cloud as ply...', self.xyz.shape)
        vertex = np.concatenate([self.xyz,self.color,self.conf,self.dirx,self.diry,self.dirz,self.embeding],axis=-1)
        vertex = [tuple(i) for i in vertex]
            #ply的格式，没写循环、增加可读性
        vertex = np.array(vertex,
                dtype=[
                    ("x", np.dtype("float32")),
                    ("y", np.dtype("float32")),
                    ("z", np.dtype("float32")),
                    ("red", np.dtype("float32")),
                    ("green", np.dtype("float32")),
                    ("blue", np.dtype("float32")),
                    ("conf", np.dtype("float32")),
                    ("dirx0", np.dtype("float32")),
                    ("dirx1", np.dtype("float32")),
                    ("dirx2", np.dtype("float32")),
                    ("diry0", np.dtype("float32")),
                    ("diry1", np.dtype("float32")),
                    ("diry2", np.dtype("float32")),
                    ("dirz0", np.dtype("float32")),
                    ("dirz1", np.dtype("float32")),
                    ("dirz2", np.dtype("float32")),
                    ("embeding0", np.dtype("float32")),
                    ("embeding1", np.dtype("float32")),
                    ("embeding2", np.dtype("float32")),
                    ("embeding3", np.dtype("float32")),
                    ("embeding4", np.dtype("float32")),
                    ("embeding5", np.dtype("float32")),
                    ("embeding6", np.dtype("float32")),
                    ("embeding7", np.dtype("float32")),
                    ("embeding8", np.dtype("float32")),
                    ("embeding9", np.dtype("float32")),
                    ("embeding10", np.dtype("float32")),
                    ("embeding11", np.dtype("float32")),
                    ("embeding12", np.dtype("float32")),
                    ("embeding13", np.dtype("float32")),
                    ("embeding14", np.dtype("float32")),
                    ("embeding15", np.dtype("float32")),
                    ("embeding16", np.dtype("float32")),
                    ("embeding17", np.dtype("float32")),
                    ("embeding18", np.dtype("float32")),
                    ("embeding19", np.dtype("float32")),
                    ("embeding20", np.dtype("float32")),
                    ("embeding21", np.dtype("float32")),
                    ("embeding22", np.dtype("float32")),
                    ("embeding23", np.dtype("float32")),
                    ("embeding24", np.dtype("float32")),
                    ("embeding25", np.dtype("float32")),
                    ("embeding26", np.dtype("float32")),
                    ("embeding27", np.dtype("float32")),
                    ("embeding28", np.dtype("float32")),
                    ("embeding29", np.dtype("float32")),
                    ("embeding30", np.dtype("float32")),
                    ("embeding31", np.dtype("float32")),
                ]
            )
        ply_pc = PlyElement.describe(vertex, "vertex")
        ply_pc = PlyData([ply_pc])
        self.save_ply_path = os.path.join(self.file_dir, name + '_penerfneuralpoint.ply')
        ply_pc.write(self.save_ply_path)
        print('Save done')

    def select_from_meshlabpoint(self,meshlab_point):
        id_list = find_neural_point_id(meshlab_point.xyz,self.xyz)
        neural_ptr = create_neural_point(self.opt,'penerf')
        neural_ptr.set_input(self.xyz[id_list],points_embeding=self.embeding[id_list],points_conf=self.conf[id_list],points_dirx = self.dirx[id_list],points_diry=self.diry[id_list],points_dirz=self.dirz[id_list],points_color=self.color[id_list])
        return neural_ptr

    def __sub__(self, other:BaseNeuralPoint):
        id_list = find_neural_point_id( other.xyz,self.xyz)
        all_id_list = np.arange(len(self.xyz))
        not_id_list = list(set(all_id_list)-set(id_list))
        neural_ptr = create_neural_point(self.opt, 'penerf')
        neural_ptr.set_input(self.xyz[not_id_list], points_embeding=self.embeding[not_id_list],
                             points_conf=self.conf[not_id_list], points_dirx=self.dirx[not_id_list],
                             points_diry=self.diry[not_id_list], points_dirz=self.dirz[not_id_list],
                             points_color=self.color[not_id_list])
        return neural_ptr

    def __add__(self, other):
        new_ptr = create_neural_point(self.opt, 'penerf')
        new_xyz = np.concatenate((self.xyz, other.xyz), axis=0)
        new_color = np.concatenate((self.color, other.color), axis=0)
        new_embeding = np.concatenate((self.embeding, other.embeding), axis=0)
        new_conf = np.concatenate((self.conf, other.conf), axis=0)
        new_dirx = np.concatenate((self.dirx, other.dirx), axis=0)
        new_diry = np.concatenate((self.diry, other.diry), axis=0)
        new_dirz = np.concatenate((self.dirz, other.dirz), axis=0)
        new_ptr.set_input(new_xyz, points_embeding=new_embeding,
                             points_conf=new_conf, points_dirx=new_dirx,
                             points_diry=new_diry, points_dirz=new_dirz,
                             points_color=new_color)
        return new_ptr

    def translate(self, translation_matrix: np.ndarray = None, rotate_itself=False,rotate_centerpoint=np.array([0, 0, 0])):
        '''
        translation_matrix:4*4
        rotate_centerpoint:as default rotate by world coord.
        If set rotate_itself=True, it will set rotate_centerpoint as the barycentre of the object
        '''
        assert translation_matrix.shape == (4, 4) and type(
            translation_matrix) == np.ndarray, 'Please input a translation matrix of 4*4'
        new_ptr = deepcopy(self)
        if rotate_itself==True:
            rotate_centerpoint = np.sum(new_ptr.xyz, axis=0) / new_ptr.xyz.shape[0]
        rot_matrix = translation_matrix[:3, :3]
        trans_vector = translation_matrix[:3, 3]
        new_ptr.xyz = (new_ptr.xyz - rotate_centerpoint) @ rot_matrix + trans_vector + rotate_centerpoint
        new_ptr.dirx = new_ptr.dirx @ rot_matrix
        new_ptr.diry = new_ptr.diry @ rot_matrix
        new_ptr.dirz = new_ptr.dirz @ rot_matrix
        return new_ptr

    def down_sample(self,voxel_size = np.array([0.003,0.003,0.003]),max_ptr_in_a_occ = 4,down_sample_func='random'):
        lower = np.min(self.xyz, axis=0)
        upper = np.max(self.xyz, axis=0)
        bbox = np.concatenate([lower[..., None], upper[..., None]], axis=-1)
        occ_shift = bbox[:,0]-1e-9
        xyz = deepcopy(self.xyz)
        embeding = deepcopy(self.embeding)
        color = deepcopy(self.color)
        conf = deepcopy(self.conf)
        dirx = deepcopy(self.dirx)
        diry = deepcopy(self.diry)
        dirz = deepcopy(self.dirz)
        shifted_neural_ptr = xyz - occ_shift
        shifted_neural_ptr_occ_coord = np.floor(shifted_neural_ptr / voxel_size).astype(int)  # [n,3] int
        occ_shape = np.ceil((bbox[:, 1] - bbox[:, 0]) / voxel_size).astype(int)
        occ_index = np.zeros([occ_shape[0], occ_shape[1], occ_shape[2], max_ptr_in_a_occ],
                             dtype=np.int32)  # e.g[60,50,40,32]
        occ_mask = np.zeros(occ_shape, dtype=np.int32)
        # build occ
        for i in range(len(shifted_neural_ptr)):
            coord_x, coord_y, coord_z = shifted_neural_ptr_occ_coord[i]  # [3,]
            pos =  occ_mask[coord_x, coord_y, coord_z]
            if pos < max_ptr_in_a_occ-1:
                occ_index[coord_x, coord_y, coord_z, pos] = i
                occ_mask[coord_x, coord_y, coord_z] += 1
            else :
                print('warning!! out range, max_ptr_in_a_occ not enough')
        # quick search
        down_sampled_xyz  = np.zeros([0, 3],dtype=np.float32)
        down_sampled_embeding = np.zeros([0, 32], dtype=np.float32)
        down_sampled_conf = np.zeros([0, 1], dtype=np.float32)
        down_sampled_color = np.zeros([0, 3], dtype=np.float32)
        down_sampled_dirx = np.zeros([0, 3], dtype=np.float32)
        down_sampled_diry = np.zeros([0, 3], dtype=np.float32)
        down_sampled_dirz = np.zeros([0, 3], dtype=np.float32)

        xyz_reshape = xyz[occ_index]
        print('xyz_reshape:{}'.format(xyz_reshape.shape))
        embeding_reshape = embeding[occ_index]
        print('embeding_reshape:{}'.format(embeding_reshape.shape))
        color_reshape = color[occ_index]
        conf_reshape = conf[occ_index]
        print('conf_reshape:{}'.format(conf_reshape.shape))
        dirx_reshape = dirx[occ_index]
        diry_reshape = diry[occ_index]
        dirz_reshape = dirz[occ_index]
        print('dir_reshape:{}'.format(dirz_reshape.shape))

        if down_sample_func == 'random':
            for h in tqdm(range(occ_mask.shape[0])):
                for w in range(occ_mask.shape[1]):
                    for c in range(occ_mask.shape[2]):
                        index = occ_mask[h][w][c]
                        if index!=0:
                            tmp_xyz = xyz_reshape[h][w][c][index-1]
                            tmp_embeding = embeding_reshape[h][w][c][index - 1]
                            tmp_conf = conf_reshape[h][w][c][index - 1]
                            tmp_dirx = dirx_reshape[h][w][c][index - 1]
                            tmp_diry = diry_reshape[h][w][c][index - 1]
                            tmp_dirz = dirz_reshape[h][w][c][index - 1]
                            tmp_color = color_reshape[h][w][c][index - 1]
                            down_sampled_xyz = np.concatenate([down_sampled_xyz,tmp_xyz[None,...]],axis=0)
                            down_sampled_embeding = np.concatenate([down_sampled_embeding, tmp_embeding[None, ...]], axis=0)
                            down_sampled_conf = np.concatenate([down_sampled_conf, tmp_conf[None, ...]], axis=0)
                            down_sampled_dirx = np.concatenate([down_sampled_dirx, tmp_dirx[None, ...]], axis=0)
                            down_sampled_diry = np.concatenate([down_sampled_diry, tmp_diry[None, ...]], axis=0)
                            down_sampled_dirz = np.concatenate([down_sampled_dirz, tmp_dirz[None, ...]], axis=0)
                            down_sampled_color = np.concatenate([down_sampled_color, tmp_color[None, ...]], axis=0)
        if down_sample_func == 'average':
            print(xyz_reshape.shape)
            xyz_reshape_cumsum = np.cumsum( xyz_reshape,axis = 3)
            embeding_reshape_cumsum = np.cumsum(embeding_reshape, axis=3)
            conf_reshape_cumsum = np.cumsum(conf_reshape, axis=3)
            dirx_reshape_cumsum = np.cumsum(dirx_reshape, axis=3)
            diry_reshape_cumsum = np.cumsum(diry_reshape, axis=3)
            dirz_reshape_cumsum = np.cumsum(dirz_reshape, axis=3)
            color_reshape_cumsum = np.cumsum(color_reshape, axis=3)
            for h in tqdm(range(occ_mask.shape[0])):
                for w in range(occ_mask.shape[1]):
                    for c in range(occ_mask.shape[2]):
                        index = occ_mask[h][w][c]
                        if index!=0:
                            tmp_xyz = xyz_reshape_cumsum[h][w][c][index-1]/index
                            tmp_embeding = embeding_reshape_cumsum[h][w][c][index - 1]/index
                            tmp_conf = conf_reshape_cumsum[h][w][c][index - 1]/index
                            tmp_dirx = dirx_reshape_cumsum[h][w][c][index - 1]/index
                            tmp_diry = diry_reshape_cumsum[h][w][c][index - 1]/index
                            tmp_dirz = dirz_reshape_cumsum[h][w][c][index - 1]/index
                            tmp_color = color_reshape_cumsum[h][w][c][index - 1] / index
                            down_sampled_xyz = np.concatenate([down_sampled_xyz, tmp_xyz[None, ...]], axis=0)
                            down_sampled_embeding = np.concatenate([down_sampled_embeding, tmp_embeding[None, ...]],axis=0)
                            down_sampled_conf = np.concatenate([down_sampled_conf, tmp_conf[None, ...]], axis=0)
                            down_sampled_dirx = np.concatenate([down_sampled_dirx, tmp_dirx[None, ...]], axis=0)
                            down_sampled_diry = np.concatenate([down_sampled_diry, tmp_diry[None, ...]], axis=0)
                            down_sampled_dirz = np.concatenate([down_sampled_dirz, tmp_dirz[None, ...]], axis=0)
                            down_sampled_color = np.concatenate([down_sampled_color, tmp_color[None, ...]], axis=0)
        new_ptr = deepcopy(self)
        new_ptr.xyz = down_sampled_xyz
        new_ptr.color = down_sampled_color
        new_ptr.embeding = down_sampled_embeding
        new_ptr.conf = down_sampled_conf
        new_ptr.dirx = down_sampled_dirx
        new_ptr.diry = down_sampled_diry
        new_ptr.dirz = down_sampled_dirz
        print(down_sampled_xyz)
        return new_ptr

