import os
from plyfile import PlyData, PlyElement
import numpy as np
from copy import deepcopy
from utils_mine.occ2ptr2occ import build_occ
class BasePoint:
    def __init__(self,opt):
        self.file_dir = os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans)
        self.opt = opt
        self.xyz = None

    def set_input(self,points_xyz,**kwargs):
        self.xyz = points_xyz

    def load_from_ply(self,name):
        raise NotImplementedError

    def save_as_ply(self,name):
        raise NotImplementedError

    def change_scale(self,scale_factor):
        '''
        if scale_factor <1 scale down ,if scale_factor >1 scale up
        attention:scale up/down with the center of barycentre of the object
        '''
        new_ptr = deepcopy(self)
        barycenter = np.sum(new_ptr.xyz, axis=0) / self.xyz.shape[0]
        new_ptr.xyz = (new_ptr.xyz-barycenter)*scale_factor+barycenter
        return new_ptr

    def down_sample(self,voxel_size = np.array([0.008,0.008,0.008]),max_ptr_in_a_occ = 64,is_interpolate_feature=False):
        raise NotImplementedError