import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np
from Editor.pointcloud import *
#np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

import cv2
from tqdm import tqdm

class PointCloudEditor:
    def __init__(self,opt):
        self.opt = opt
        self.if_has_semantic_label = opt.has_semantic_label
        pass
    def crop_point_cloud(self,npcd_child,npcd_father):# remove npt2 point from npt1,return removed npt1
        '''
        Actually i dont know how to do that...
        I use every point in meshlabpcd to find the nearest pcd in neural_pcd
        '''
        # TODO: use cuda to reimplementation
        npc = Neural_pointcloud(self.opt)
        pointsize_child =npcd_child.xyz.shape[0]
        pointsize_father = npcd_father.xyz.shape[0]
        neural_xyz = np.empty([pointsize_father,3])
        neural_color = np.empty([pointsize_father,3])
        neural_embeding = np.empty([pointsize_father,32])
        neural_conf = np.empty([pointsize_father])
        if self.if_has_semantic_label:
            neural_label = np.empty([pointsize_father])
        else :
            neural_label = None
        neural_dirx = np.empty([pointsize_father,3])
        neural_diry = np.empty([pointsize_father, 3])
        neural_dirz = np.empty([pointsize_father, 3])
        print('Scale of father neural point cloud :',(pointsize_father))
        print('Scale of child neural point cloud:',(pointsize_child))
        idx = 0
        for i in tqdm(range(pointsize_father)):
            father_ptr_xyz = npcd_father.xyz[i]
            dis = np.sqrt(np.sum(np.square(father_ptr_xyz-npcd_child.xyz),axis = -1))
            if  not (dis < 1e-7).any():
                neural_xyz[idx] = npcd_father.xyz[i]
                neural_color[idx] = npcd_father.color[i]
                neural_embeding[idx] = npcd_father.embeding[i]
                neural_conf[idx] = npcd_father.conf[i]
                if self.if_has_semantic_label:
                    neural_label[idx] = npcd_father.label[i]
                neural_dirx[idx] = npcd_father.dirx[i]
                neural_diry[idx] = npcd_father.diry[i]
                neural_dirz[idx] = npcd_father.dirz[i]
                idx+=1
        neural_xyz = neural_xyz[:idx]
        neural_color = neural_color[:idx]
        neural_embeding = neural_embeding[:idx]
        neural_conf = neural_conf[:idx]
        neural_dirx = neural_dirx[:idx]
        neural_diry = neural_diry[:idx]
        neural_dirz = neural_dirz[:idx]
        if self.if_has_semantic_label:
            neural_label = neural_label[:idx]
        print('\ncrop done...neural point cloud scale:',idx)
        npc.load_from_var(neural_xyz,neural_embeding,neural_conf,neural_dirx,neural_diry,neural_dirz,neural_color,neural_label)
        return npc
    def translation_point_cloud_global(self,npcd, transMatirx,centerpoint=np.array([0,0,0])):#rotate by world coordinate
        res_npc = Neural_pointcloud(self.opt)
        rot_matrix = transMatirx[:3, :3]
        trans_vector = transMatirx[:3, 3]
        res_npc.xyz = (npcd.xyz-centerpoint) @ rot_matrix + trans_vector + centerpoint
        res_npc.color = npcd.color
        res_npc.embeding = npcd.embeding
        res_npc.conf = npcd.conf
        res_npc.dirx = npcd.dirx@ rot_matrix
        res_npc.diry = npcd.diry@ rot_matrix
        res_npc.dirz = npcd.dirz @ rot_matrix
        if self.if_has_semantic_label:
            res_npc.label = npcd.label
        return res_npc
    def translation_point_cloud_local(self,npcd,transMatirx):#rotate by self coordinate
        pointsize = npcd.xyz.shape[0]
        centerptr = np.sum(npcd.xyz,axis=0)/pointsize
        res_npc = Neural_pointcloud(self.opt)
        rot_matrix = transMatirx[:3, :3]
        trans_vector = transMatirx[:3, 3]
        res_npc.xyz = (npcd.xyz-centerptr) @ rot_matrix + trans_vector + centerptr
        res_npc.color = npcd.color
        res_npc.embeding = npcd.embeding
        res_npc.conf = npcd.conf
        res_npc.dirx = npcd.dirx@ rot_matrix
        res_npc.diry = npcd.diry@ rot_matrix
        res_npc.dirz = npcd.dirz @ rot_matrix
        if self.if_has_semantic_label:
            res_npc.label = npcd.label
        return res_npc
    def add_point_cloud(self,npcd_child,npcd_father):
        res_npc = Neural_pointcloud(self.opt)
        res_npc.xyz = np.concatenate((npcd_child.xyz,npcd_father.xyz),axis = 0)
        res_npc.color = np.concatenate((npcd_child.color, npcd_father.color), axis=0)
        res_npc.embeding = np.concatenate((npcd_child.embeding, npcd_father.embeding), axis=0)
        res_npc.conf = np.concatenate((npcd_child.conf, npcd_father.conf), axis=0)
        res_npc.dirx = np.concatenate((npcd_child.dirx, npcd_father.dirx), axis=0)
        res_npc.diry = np.concatenate((npcd_child.diry, npcd_father.diry), axis=0)
        res_npc.dirz = np.concatenate((npcd_child.dirz, npcd_father.dirz), axis=0)
        if self.if_has_semantic_label:
            res_npc.label = np.concatenate((npcd_child.label, npcd_father.label), axis=0)
        return res_npc