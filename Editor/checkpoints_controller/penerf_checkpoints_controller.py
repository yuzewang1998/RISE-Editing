from Editor.checkpoints_controller.base_checkpoints_controller import  BaseCheckpointsController
from Editor.points import create_neural_point
import torch
import os
import numpy as np
class PeNerfCheckpointsController(BaseCheckpointsController):
    def __init__(self,opt,name):
         super().__init__(opt,name)
    def cvt_2_neuralPoint(self):
        super().cvt_2_neuralPoint()
        self.points_dirx = self.network_paras["neural_points.points_dirx"].view(-1, 3).cpu().numpy()
        self.points_diry = self.network_paras["neural_points.points_diry"].view(-1, 3).cpu().numpy()
        self.points_dirz = self.network_paras["neural_points.points_dirz"].view(-1, 3).cpu().numpy()
        print('point cloud scale:', self.points_xyz.shape, type(self.points_xyz))
        neural_point = create_neural_point(self.opt,'penerf')
        neural_point.set_input(self.points_xyz,points_embeding=self.points_embeding,points_conf=self.points_conf,points_color=self.points_color,points_dirx=self.points_dirx,points_diry=self.points_diry,points_dirz=self.points_dirz)
        return neural_point
    def set_and_save(self,penerf_neuralpoint = None,edit_name = None):
        print('Saving checkpoints from neural point cloud...')
        if penerf_neuralpoint is not None:
            self.network_paras["neural_points.xyz"] = torch.Tensor(penerf_neuralpoint.xyz)  #[ptr,3]
            self.network_paras["neural_points.points_embeding"] = torch.unsqueeze(torch.Tensor(penerf_neuralpoint.embeding),dim=0) #[1,ptr,32]
            self.network_paras["neural_points.points_conf"] =  torch.unsqueeze(torch.Tensor(penerf_neuralpoint.conf[...,np.newaxis]),dim=0)#[1,ptr,1]
            self.network_paras["neural_points.points_dirx"] = torch.unsqueeze(torch.Tensor(penerf_neuralpoint.dirx),dim=0)#[1,ptr,3]
            self.network_paras["neural_points.points_diry"] = torch.unsqueeze(torch.Tensor(penerf_neuralpoint.diry),dim=0)#[1,ptr,3]
            self.network_paras["neural_points.points_dirz"] = torch.unsqueeze(torch.Tensor(penerf_neuralpoint.dirz), dim=0)  # [1,ptr,3]
            self.network_paras["neural_points.points_color"] = torch.unsqueeze(torch.Tensor(penerf_neuralpoint.color),dim=0) #[1,ptr,3]
        torch.save(self.network_paras,os.path.join(self.opt.editor_checkpoints_root,self.opt.editor_checkpoints_scans,self.checkpoints_name+'_'+edit_name +'.pth'))# find the latest pth file)
        print('Saving checkpoints done')
    def aggrator_paras_copy(self,cpc_other):
        for key in cpc_other.network_paras.keys():
            if not key.startswith('neural_points'):
                self.network_paras[key] = cpc_other.network_paras[key]

