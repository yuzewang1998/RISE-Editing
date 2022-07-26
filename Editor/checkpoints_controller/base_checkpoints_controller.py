import os
import torch
class BaseCheckpointsController:
    def __init__(self,opt,name):
        self.opt = opt
        self.checkpoints_name = name
        self.checkpoints_path = os.path.join(opt.checkpoints_root,opt.checkpoints_scans,self.checkpoints_name +'.pth')
        self.network_paras = torch.load(self.checkpoints_path, map_location=torch.device('cpu'))
    def cvt_2_neuralPoint(self):
        self.points_xyz = self.network_paras["neural_points.xyz"].view(-1, 3).cpu().numpy()
        self.points_embeding = self.network_paras["neural_points.points_embeding"].view(-1, 32).cpu().numpy()
        self.points_conf = self.network_paras["neural_points.points_conf"].view(-1, 1).cpu().numpy()
        self.points_color = self.network_paras["neural_points.points_color"].view(-1, 3).cpu().numpy()
    def set_and_save(self,points,name):
        raise NotImplementedError
