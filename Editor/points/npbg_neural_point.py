from Editor.points.base_point import BasePoint
from Editor.points.base_neural_point import BaseNeuralPoint
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from Editor.points import create_neural_point,find_neural_point_id
from copy import deepcopy

class NpbgNeuralPoint(BasePoint):
    def __init__(self,opt):
        super().__init__(opt)
        self.embeding = None

    def set_input(self, points_xyz, **kwargs):
        super().set_input(points_xyz)
        self.embeding = kwargs['points_embeding']

    def load_from_ply(self, name):
        self.load_ply_path = os.path.join(self.file_dir, name + '_npbgneuralpoint.ply')
        print(self.load_ply_path)
        assert os.path.exists(self.load_ply_path), '{}_npbgneuralpoint doesn`t exist ,check!'.format(name)
        print('loading neural point cloud from ply....')
        plydata = PlyData.read(self.load_ply_path)
        x, y, z = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
            plydata.elements[0].data["y"].astype(np.float32)), np.array(
            plydata.elements[0].data["z"].astype(np.float32))
        self.xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)
        self.embeding = np.zeros([self.xyz.shape[0],0])
        for i in range(56):
            tmp = np.array(plydata.elements[0].data["embeding"+str(i)].astype(np.float32))[...,np.newaxis]
            self.embeding =np.concatenate([self.embeding,tmp],axis = -1)
        print('loading done. Scale of neural point cloud:',self.embeding.shape[0])

    def save_as_ply(self,name):
        assert self.xyz is not None, '[ERROR]Save before load,check it!'
        print('Saving neural point cloud as ply...', self.xyz.shape)
        vertex = np.concatenate([self.xyz,self.embeding],axis=-1)
        vertex = [tuple(i) for i in vertex]
            #ply的格式，没写循环、增加可读性
        vertex = np.array(vertex,
                dtype=[
                    ("x", np.dtype("float32")),
                    ("y", np.dtype("float32")),
                    ("z", np.dtype("float32")),
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
                    ("embeding32", np.dtype("float32")),
                    ("embeding33", np.dtype("float32")),
                    ("embeding34", np.dtype("float32")),
                    ("embeding35", np.dtype("float32")),
                    ("embeding36", np.dtype("float32")),
                    ("embeding37", np.dtype("float32")),
                    ("embeding38", np.dtype("float32")),
                    ("embeding39", np.dtype("float32")),
                    ("embeding40", np.dtype("float32")),
                    ("embeding41", np.dtype("float32")),
                    ("embeding42", np.dtype("float32")),
                    ("embeding43", np.dtype("float32")),
                    ("embeding44", np.dtype("float32")),
                    ("embeding45", np.dtype("float32")),
                    ("embeding46", np.dtype("float32")),
                    ("embeding47", np.dtype("float32")),
                    ("embeding48", np.dtype("float32")),
                    ("embeding49", np.dtype("float32")),
                    ("embeding50", np.dtype("float32")),
                    ("embeding51", np.dtype("float32")),
                    ("embeding52", np.dtype("float32")),
                    ("embeding53", np.dtype("float32")),
                    ("embeding54", np.dtype("float32")),
                    ("embeding55", np.dtype("float32")),
                ]
            )
        ply_pc = PlyElement.describe(vertex, "vertex")
        ply_pc = PlyData([ply_pc])
        self.save_ply_path = os.path.join(self.file_dir, name + '_npbgneuralpoint.ply')
        ply_pc.write(self.save_ply_path)
        print('Save done')

    def select_from_meshlabpoint(self,meshlab_point):
        id_list = find_neural_point_id(meshlab_point.xyz,self.xyz)
        neural_ptr = create_neural_point(self.opt,'npbg')
        neural_ptr.set_input(self.xyz[id_list],points_embeding=self.embeding[id_list])
        return neural_ptr

    def __sub__(self, other:BaseNeuralPoint):
        id_list = find_neural_point_id( other.xyz,self.xyz)
        all_id_list = np.arange(len(self.xyz))
        not_id_list = list(set(all_id_list)-set(id_list))
        neural_ptr = create_neural_point(self.opt, 'npbg')
        neural_ptr.set_input(self.xyz[not_id_list], points_embeding=self.embeding[not_id_list])
        return neural_ptr

    def __add__(self, other):
        new_ptr = create_neural_point(self.opt, 'npbg')
        new_xyz = np.concatenate((self.xyz, other.xyz), axis=0)
        new_embeding = np.concatenate((self.embeding, other.embeding), axis=0)
        new_ptr.set_input(new_xyz, points_embeding=new_embeding)
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
