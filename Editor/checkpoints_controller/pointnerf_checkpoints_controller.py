from Editor.checkpoints_controller.base_checkpoints_controller import  BaseCheckpointsController
from Editor.points import create_neural_point
import torch
import numpy as np
import os
class PointNerfCheckpointsController(BaseCheckpointsController):
    def __init__(self,opt,name):
        super().__init__(opt,name)
    def cvt_2_neuralPoint(self):
        super().cvt_2_neuralPoint()
        self.points_dir = self.network_paras["neural_points.points_dir"].view(-1, 3).cpu().numpy()
        print('point cloud scale:', self.points_xyz.shape, type(self.points_xyz))
        neural_point = create_neural_point(self.opt,'pointnerf')
        neural_point.set_input(self.points_xyz,points_embeding=self.points_embeding,points_conf=self.points_conf,points_color=self.points_color,points_dir=self.points_dir)
        return neural_point
    def set_and_save(self,pointnerf_neuralpoint=None,edit_name = None,ani_flag = 0):
        print('Saving checkpoints from neural point cloud...')
        if pointnerf_neuralpoint is not None:
            self.network_paras["neural_points.xyz"] = torch.Tensor(pointnerf_neuralpoint.xyz)  #[ptr,3]
            self.network_paras["neural_points.points_embeding"] = torch.unsqueeze(torch.Tensor(pointnerf_neuralpoint.embeding),dim=0) #[1,ptr,32]
            self.network_paras["neural_points.points_conf"] =  torch.unsqueeze(torch.Tensor(pointnerf_neuralpoint.conf[...,np.newaxis]),dim=0)#[1,ptr,1]
            self.network_paras["neural_points.points_dir"] = torch.unsqueeze(torch.Tensor(pointnerf_neuralpoint.dir),dim=0)#[1,ptr,3]
            self.network_paras["neural_points.points_color"] = torch.unsqueeze(torch.Tensor(pointnerf_neuralpoint.color),dim=0) #[1,ptr,3]
            if not ani_flag:
                if edit_name != None:
                    torch.save(self.network_paras,
                               os.path.join(self.opt.editor_checkpoints_root, self.opt.editor_checkpoints_scans,
                                            self.checkpoints_name + '_' + edit_name + '.pth'))  # find the latest pth file)
                else:
                    torch.save(self.network_paras,
                               os.path.join(self.opt.editor_checkpoints_root, self.opt.editor_checkpoints_scans,
                                            self.checkpoints_name + '.pth'))
            else:
                torch.save(self.network_paras,
                           os.path.join(os.path.join(self.opt.editor_checkpoints_root,self.opt.editor_checkpoints_scans), edit_name + '_net_ray_marching.pth'))
                torch.save(self.network_paras,
                           os.path.join(self.opt.editor_checkpoints_root, edit_name + '_net_ray_marching.pth'))
                print('Saving checkpoints done: {}'.format(
                    os.path.join(self.opt.editor_checkpoints_root, edit_name + '_net_ray_marching.pth')))
            # torch.save(self.network_paras,os.path.join(self.opt.editor_checkpoints_root,self.opt.editor_checkpoints_scans,self.checkpoints_name+'_'+edit_name +'.pth'))# find the latest pth file)
        print('Saving checkpoints done')
