from re import L
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from Editor.checkpoints_controller import create_checkpointscontroller
from options import TrainOptions

from run import train_ft_ms

from asyncio import subprocess
import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))




class CPC_Paramter:
    editor_checkpoints_root = ""
    editor_checkpoints_scans = ""


def main():


    # 在这里定义有几个场景，进行多次训练
    TO = TrainOptions()
    opt = TO.parse()
    swap_num = opt.swap_num
    train_step = opt.train_step
    #这里是整体网络的opt需要更新的option
    # filename_0 = 'lego'
    # scene0 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_0),
    #     "scan": filename_0,
    #     "ranges": [-0.638,-1.141,-0.346,0.634,1.149,1.141] ,
    #     "train_step": train_step,
    #     "renderer_required_grad": 1,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_0}
    # }
    # filename_1 = 'chair'
    # scene1 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_1),
    #     "scan": filename_1,
    #     "ranges": [-0.721,-0.695,-0.995,0.658,0.706,1.050],
    #     "train_step": train_step,
    #     "renderer_required_grad": 1,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_1}
    # }
    # filename_2 = 'mic'
    # scene2 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_2),
    #     "scan": filename_2,
    #     "ranges": [-1.252, -0.910 ,-0.742 ,0.767 ,1.082 ,1.151 ],
    #     "train_step": train_step,
    #     "renderer_required_grad": 1,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_2}
    # }
    # filename_3 = 'hotdog'
    # scene3 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_3),
    #     "scan": filename_3,
    #     "ranges": [ -1.198 ,-1.286, -0.190,  1.198, 1.110, 0.312],
    #     "train_step": train_step,
    #     "renderer_required_grad": 0,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_3}
    # }
    # filename_4 = 'drums'
    # scene4 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_4),
    #     "scan": filename_4,
    #     "ranges": [ -1.126, -0.746 ,-0.492, 1.122 ,0.962 ,0.939 ],
    #     "train_step": train_step,
    #     "renderer_required_grad": 1,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_4}
    # }
    # filename_5 = 'ficus'
    # scene5 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_5),
    #     "scan": filename_5,
    #     "ranges": [ -0.377, -0.858, -1.034, 0.555 ,0.578 ,1.141 ],
    #     "train_step": train_step,
    #     "renderer_required_grad": 1,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_5}
    # }
    # filename_6 = 'materials'
    # scene6 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_6),
    #     "scan": filename_6,
    #     "ranges": [ -1.123 , -0.759 ,-0.232 , 1.072 ,0.986, 0.200  ],
    #     "train_step": train_step,
    #     "renderer_required_grad": 1,
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_6}
    # }
    # scene_list = {
    #     0: scene0,
    #     1: scene1,
    #     2: scene2,
    #     3: scene5,
    #     4: scene6,
    # }

    scannet_nerf_lego = {
        "data_root" : "../data_src/nerf/nerf_synthetic_colmap/",
        'scan' : "lego",
        'dataset_name' : 'nerf_synth360_ft',
        'name': 'train_multi_scene/lego',
        'P': 13,
        'SR': 80,
        'default_conf': 0.15,
        'depth_occ': 1,
        'random_sample_size': 60,
        'vsize': [0.004, 0.004, 0.004],
        'zero_one_loss_items': [],
        'zero_one_loss_weights': [1.0],
        'train_step':500,
        "cpc_opt": {'editor_checkpoints_root': os.path.join(opt.checkpoints_dir, opt.name.split('/')[0]),
                    'editor_checkpoints_scans': 'lego'}
    }
    scannet_nerf_chair = {
        "data_root" : "../data_src/nerf/nerf_synthetic_colmap/",
        'scan' : "chair",
        'dataset_name' : 'nerf_synth360_ft',
        'name': 'train_multi_scene/chair',
        'P': 13,
        'SR': 80,
        'default_conf': 0.15,
        'depth_occ': 1,
        'random_sample_size': 60,
        'vsize': [0.004, 0.004, 0.004],
        'zero_one_loss_items': [],
        'zero_one_loss_weights': [1.0],
        'train_step':500,
        "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
                   'editor_checkpoints_scans':'chair'}

    }

    scannet_nerf_sc113 = {
        "data_root": "../data_src/scannet/scans/",
        'scan': "scene0113_00",
        'dataset_name': 'scannet_ft',
        'name': 'train_multi_scene/sc113',
        'P': 32,
        'SR': 24,
        'default_conf':-1.0,
        'depth_occ':0,
        'random_sample_size':32,
        'vsize':[0.008,0.008,0.008],
        'zero_one_loss_items': ['conf_coefficient'],
        'zero_one_loss_weights': [0.0001],
        'train_step':1000,
        "cpc_opt": {'editor_checkpoints_root': os.path.join(opt.checkpoints_dir, opt.name.split('/')[0]),
                    'editor_checkpoints_scans': 'sc113'}
    }

    scene_list = {
        0: scannet_nerf_sc113,
        1: scannet_nerf_lego,
        2: scannet_nerf_chair,
    }
    sceneListSize = len(scene_list)
    #这个opt应该是cpc的opt
    cpc_para_list = []
    for i in range(sceneListSize):
        para_t = CPC_Paramter()
        para_t.editor_checkpoints_root = scene_list[i]["cpc_opt"]['editor_checkpoints_root']
        para_t.editor_checkpoints_scans = scene_list[i]["cpc_opt"]['editor_checkpoints_scans']
        cpc_para_list.append(para_t)

    if swap_num < sceneListSize:
        # 每个场景先训练一个基本的
        print("swapNum:", swap_num)
        print("trainStep:",train_step)
        update_opt = scene_list[swap_num]
        train_ft_ms.train_one_scene(update_opt)
    else:
        baseInd = swap_num % sceneListSize
        nextInd = (swap_num + 1) % sceneListSize

        cpc_para_base = cpc_para_list[baseInd]
        cpc_para_next = cpc_para_list[nextInd]
        cpcbase = create_checkpointscontroller(cpc_para_base,'penerf',None)
        cpcnext = create_checkpointscontroller(cpc_para_next, 'penerf', None)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("OutswapNum:", swap_num)
        print("trainStep:",train_step)
        print("base:", cpcbase.checkpoints_path)
        print("next", cpcnext.checkpoints_path)
        print("要训练：", cpcnext.checkpoints_path)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # 把cpcbase的参数赋值给cpcnext
        cpcnext.aggrator_paras_copy(cpcbase)
        cpcnext.set_and_save(penerf_neuralpoint=None,edit_name=None)
        update_opt = scene_list[nextInd]
        # if (swap_num % 100 > 20) or swap_num>1000 :
        #     update_opt['renderer_required_grad'] = 1
        # else:
        #     update_opt['renderer_required_grad'] = 1
        # update_opt['renderer_required_grad'] = 0
        train_ft_ms.train_one_scene(update_opt)

if __name__ == '__main__':
    main()
