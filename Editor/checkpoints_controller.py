import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np
#np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from Editor.pointcloud import  *
import cv2
from tqdm import tqdm

class CheckpointsController:
    def __init__(self,opt):
        self.opt = opt
        self.points_embeding_dim = opt.points_feature_dim
        self.if_has_semantic_label = opt.has_semantic_label
        self.edit_dir = os.path.join(opt.checkpoints_root,'edit')
        if 'edit' not in os.listdir(opt.checkpoints_root):
            os.makedirs(os.path.join(self.edit_dir))
        self.checkpoints_root = opt.checkpoints_root
        self.device = 'cuda:'+opt.gpu_ids
        pth_files_dir = [f for f in os.listdir(self.checkpoints_root) if f.endswith('_net_ray_marching.pth')]
        assert len(pth_files_dir)>0,'Empty checkpoints file'
        numlist = [int(i.split('_')[0]) for i in pth_files_dir]
        numlist.sort()
        self.latest_iters = str(numlist[-1])
        self.latest_file_path = os.path.join(self.checkpoints_root,self.latest_iters+'_net_ray_marching.pth')# find the latest pth file
    # find the latest pth file in root of dataset and read that file
    def load_checkpoints_as_nerualpcd(self):
        network_paras = torch.load(self.latest_file_path, map_location=torch.device('cpu'))
        print('loading checkpoints...',self.latest_file_path)
        self.points_xyz = network_paras["neural_points.xyz"].view(-1,3).cpu().numpy()
        print(self.points_xyz.shape)
        self.points_embeding = network_paras["neural_points.points_embeding"].view(-1,self.points_embeding_dim).cpu().numpy()
        self.points_conf= network_paras["neural_points.points_conf"].view(-1,1).cpu().numpy()
        self.points_dirx = network_paras["neural_points.points_dirx"].view(-1,3).cpu().numpy()
        self.points_diry = network_paras["neural_points.points_diry"].view(-1, 3).cpu().numpy()
        self.points_dirz = network_paras["neural_points.points_dirz"].view(-1, 3).cpu().numpy()
        self.points_color = network_paras["neural_points.points_color"].view(-1,3).cpu().numpy()
        if "neural_points.points_label" in network_paras.keys():
            self.points_label = network_paras["neural_points.points_label"].view(-1, 1).cpu().numpy()
        else:
            self.points_label = None
        print('point cloud scale:',self.points_color.shape,type(self.points_color))
        neural_pcd = Neural_pointcloud(self.opt)
        neural_pcd.load_from_checkpoints(self.points_xyz,self.points_embeding,self.points_conf,self.points_dirx,self.points_diry,self.points_dirz,self.points_color,self.points_label)
        return neural_pcd
    def save_checkpoints_from_neuralpcd(self,neural_pcd,name):
        print('Saving checkpoints from neural point cloud...')
        network_paras = torch.load(self.latest_file_path, map_location=self.device)
        network_paras["neural_points.xyz"] = torch.Tensor(neural_pcd.xyz)  #[ptr,3]
        network_paras["neural_points.points_embeding"] = torch.unsqueeze(torch.Tensor(neural_pcd.embeding),dim=0) #[1,ptr,32]
        network_paras["neural_points.points_conf"] =  torch.unsqueeze(torch.Tensor(neural_pcd.conf[...,np.newaxis]),dim=0)#[1,ptr,1]
        network_paras["neural_points.points_dirx"] = torch.unsqueeze(torch.Tensor(neural_pcd.dirx),dim=0)#[1,ptr,3]
        network_paras["neural_points.points_diry"] = torch.unsqueeze(torch.Tensor(neural_pcd.diry),dim=0)#[1,ptr,3]
        network_paras["neural_points.points_dirz"] = torch.unsqueeze(torch.Tensor(neural_pcd.dirz), dim=0)  # [1,ptr,3]
        network_paras["neural_points.points_color"] = torch.unsqueeze(torch.Tensor(neural_pcd.color),dim=0) #[1,ptr,3]
        if self.if_has_semantic_label :
            network_paras["neural_points.points_label"] = torch.Tensor(neural_pcd.label[...,np.newaxis])  # [ptr,3]
        torch.save(network_paras,os.path.join(self.checkpoints_root,'edit',self.latest_iters+'_net_ray_marching_Edited_'+name+'.pth'))# find the latest pth file)
        print('Saving checkpoints done')
