from Editor.points.base_neural_point  import BaseNeuralPoint
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from Editor.points.meshlab_neural_point import MeshlabNeuralPoint
from Editor.points import create_neural_point,find_neural_point_id
from copy import deepcopy
class PointnerfNeuralPoint(BaseNeuralPoint):
    def __init__(self,opt):
        super().__init__(opt)
        self.dir = None

    def set_input(self,points_xyz,**kwargs):
        super().set_input(points_xyz,**kwargs)
        self.dir = kwargs['points_dir']

    def load_from_ply(self, name):
        self.ply_path = os.path.join(self.file_dir, name + '_pointnerfneuralpoint.ply')
        assert os.path.exists(self.ply_path), '{}_pointnerfneuralpoint doesn`t exist ,check!'.format(name)
        print('loading neural point cloud from ply....')
        plydata = PlyData.read(self.ply_path)
        x, y, z = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
            plydata.elements[0].data["y"].astype(np.float32)), np.array(
            plydata.elements[0].data["z"].astype(np.float32))
        self.xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)
        r, g, b = np.array(plydata.elements[0].data["red"].astype(np.float32)), np.array(
            plydata.elements[0].data["green"].astype(np.float32)), np.array(
            plydata.elements[0].data["blue"].astype(np.float32))
        self.color = np.concatenate([r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]], axis=-1)
        dir0, dir1, dir2 = np.array(plydata.elements[0].data["dir0"].astype(np.float32)), np.array(
            plydata.elements[0].data["dir1"].astype(np.float32)), np.array(
            plydata.elements[0].data["dir2"].astype(np.float32))
        self.dir = np.concatenate([dir0[..., np.newaxis], dir1[..., np.newaxis], dir2[..., np.newaxis]], axis=-1)
        self.conf = np.array(plydata.elements[0].data["conf"].astype(np.float32))
        self.embeding = np.zeros([self.xyz.shape[0],0])
        for i in range(32):
            tmp = np.array(plydata.elements[0].data["embeding"+str(i)].astype(np.float32))[...,np.newaxis]
            self.embeding =np.concatenate([self.embeding,tmp],axis = -1)
        print('loading done. Scale of neural point cloud:',self.embeding.shape[0])

    def save_as_ply(self,name):
        assert self.xyz is not None, '[ERROR]Save before load,check it!'
        vertex = []
        print('Saving neural point cloud as ply...', self.xyz.shape)
        vertex = np.concatenate([self.xyz,self.color,self.conf,self.dir,self.embeding],axis=-1)
        vertex = [tuple(i) for i in vertex]
        vertex = np.array(
                    vertex,
                dtype=[
                    ("x", np.dtype("float32")),
                    ("y", np.dtype("float32")),
                    ("z", np.dtype("float32")),
                    ("red", np.dtype("float32")),
                    ("green", np.dtype("float32")),
                    ("blue", np.dtype("float32")),
                    ("conf", np.dtype("float32")),
                    ("dir0", np.dtype("float32")),
                    ("dir1", np.dtype("float32")),
                    ("dir2", np.dtype("float32")),
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
        self.save_ply_path = os.path.join(self.file_dir, name + '_pointnerfneuralpoint.ply')
        ply_pc.write(self.save_ply_path)
        print('Save done')

    def select_from_meshlabpoint(self,meshlab_point):
        id_list = find_neural_point_id(meshlab_point.xyz,self.xyz)
        neural_ptr = create_neural_point(self.opt,'pointnerf')
        neural_ptr.set_input(self.xyz[id_list],self.embeding[id_list],self.conf[id_list],self.dir[id_list],self.color[id_list])
        return neural_ptr

    def __sub__(self, other:BaseNeuralPoint)->BaseNeuralPoint:
        id_list = find_neural_point_id(other.xyz,self.xyz)
        all_id_list = np.arange(len(self.xyz))
        not_id_list = list(set(all_id_list)-set(id_list))
        neural_ptr = create_neural_point(self.opt, 'pointnerf')
        neural_ptr.set_input(self.xyz[not_id_list], points_embeding=self.embeding[not_id_list],
                             points_conf=self.conf[not_id_list][..., None], points_dir=self.dir,
                             points_color=self.color[not_id_list])
        return neural_ptr

    def __add__(self, other):
        new_ptr = create_neural_point(self.opt, 'pointnerf')
        new_xyz = np.concatenate((self.xyz, other.xyz), axis=0)
        new_color = np.concatenate((self.color, other.color), axis=0)
        new_embeding = np.concatenate((self.embeding, other.embeding), axis=0)
        new_conf = np.concatenate((self.conf, other.conf), axis=0)
        new_dir = np.concatenate((self.dir, other.dir), axis=0)
        new_ptr.set_input(new_xyz, points_embeding=new_embeding,
                             points_conf=new_conf, points_dir=new_dir,
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
        new_ptr.dir = new_ptr.dir @ rot_matrix
        return new_ptr

