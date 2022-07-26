from Editor.points.base_point import BasePoint
import os
from plyfile import PlyData, PlyElement
import numpy as np
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
        raise NotImplementedError
