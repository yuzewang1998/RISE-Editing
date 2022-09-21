import os
import torch
from Editor.checkpoints_controller.base_checkpoints_controller import  BaseCheckpointsController
from plyfile import PlyData, PlyElement
from Editor.points import create_neural_point
import torch
import os
import numpy as np
class NpbgCheckpointsController(BaseCheckpointsController):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--editor_checkpoints_scans',
                            type=str,
                            default='edit_something',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='scans of checkpoints')
        parser.add_argument('--editor_checkpoints_root',
                            type=str,
                            default='/home/slam/devdata/NSEPN/checkpoints/col_nerfsynth',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets')

        return parser
    def __init__(self,opt,name=None):
        self.opt = opt
        self.checkpoints_name = name
        self.checkpoints_path = os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans,self.checkpoints_name +'.pth')
        self.xyz_path = os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans,opt.xyz_name +'.ply')
        self.network_paras = torch.load(self.checkpoints_path, map_location=torch.device('cpu'))
        print("1")
    def cvt_2_neuralPoint(self):
        self.points_embeding = self.network_paras.cpu().numpy()
        plydata = PlyData.read(self.xyz_path)
        x, y, z = np.array(plydata.elements[0].data["x"].astype(np.float32)), np.array(
            plydata.elements[0].data["y"].astype(np.float32)), np.array(
            plydata.elements[0].data["z"].astype(np.float32))
        self.points_xyz = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=-1)
        neural_point = create_neural_point(self.opt,'npbg')
        neural_point.set_input(self.points_xyz,points_embeding=self.points_embeding)
        return neural_point
    def set_and_save(self,npbg_neuralpoint,edit_name=None):
        self.network_paras = torch.Tensor(npbg_neuralpoint.embeding)
        self.points_xyz = npbg_neuralpoint.xyz
        torch.save(self.network_paras,self.checkpoints_path+"n")
        vertex = np.concatenate([self.points_xyz],
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
        ply_pc.write(self.xyz_path+"1")
        print('Save done')
