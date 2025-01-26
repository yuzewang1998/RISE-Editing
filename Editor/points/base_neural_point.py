import os
from plyfile import PlyData, PlyElement
import numpy as np
from Editor.points.base_point import BasePoint
class BaseNeuralPoint(BasePoint):
    def __init__(self,opt):
        super().__init__(opt)
        self.embeding = None
        self.conf = None

    def set_input(self, points_xyz, **kwargs):
        super().set_input(points_xyz)
        self.color = kwargs['points_color']
        self.embeding = kwargs['points_embeding']
        self.conf = kwargs['points_conf']

    def load_from_ply(self,name):
        raise NotImplementedError

    def save_as_ply(self,name):
        raise NotImplementedError
    def down_sample(self,voxel_size = np.array([0.008,0.008,0.008]),max_ptr_in_a_occ = 64,is_interpolate_feature=False):
        raise NotImplementedError