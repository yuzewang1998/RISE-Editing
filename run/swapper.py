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
    filename_0 = 'lego'
    scene0 = {
        "name": os.path.join(opt.name.split('/')[0],filename_0),
        "scan": filename_0,
        "ranges": [-0.638,-1.141,-0.346,0.634,1.149,1.141] ,
        "train_step": train_step,
        "renderer_required_grad": '1',
        "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
                   'editor_checkpoints_scans':filename_0}
    }
    filename_1 = 'chair'
    scene1 = {
        "name": os.path.join(opt.name.split('/')[0],filename_1),
        "scan": filename_1,
        "ranges": [-0.721,-0.695,-0.995,0.658,0.706,1.050],
        "train_step": train_step,
        "renderer_required_grad": '1',
        "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
                   'editor_checkpoints_scans':filename_1}
    }
    # filename_2 = 'mic'
    # scene2 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_2),
    #     "scan": filename_2,
    #     "ranges": [-1.252, -0.910 ,-0.742 ,0.767 ,1.082 ,1.151 ],
    #     "train_step": train_step,
    #     "renderer_required_grad": '1',
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_2}
    # }
    # filename_3 = 'hotdog'
    # scene3 = {
    #     "name": os.path.join(opt.name.split('/')[0],filename_3),
    #     "scan": filename_3,
    #     "ranges": [ -1.198 ,-1.286, -0.190,  1.198, 1.110, 0.312],
    #     "train_step": train_step,
    #     "renderer_required_grad": '1',
    #     "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
    #                'editor_checkpoints_scans':filename_3}
    # }
    # 所有场景都放在这一个dict中
    scene_list = {
        0: scene0,
        1: scene1,
        # 2: scene2,
        # 3: scene3
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
        update_opt = scene_list[swap_num]
        train_ft_ms.train_one_scene(update_opt)
    else:
        baseInd = swap_num % sceneListSize
        nextInd = (swap_num + 1) % sceneListSize
        print("OutswapNum:", swap_num)
        cpc_para_base = cpc_para_list[baseInd]
        cpc_para_next = cpc_para_list[nextInd]
        cpcbase = create_checkpointscontroller(cpc_para_base,'penerf',None)
        cpcnext = create_checkpointscontroller(cpc_para_next, 'penerf', None)
        print("base:", cpcbase.checkpoints_path)
        print("next", cpcnext.checkpoints_path)
        print("要训练：", cpcnext.checkpoints_path)
        # 把cpcbase的参数赋值给cpcnext
        cpcnext.aggrator_paras_copy(cpcbase)
        update_opt = scene_list[nextInd]
        if (swap_num % 100 > 20) or swap_num>1000 :
            update_opt['renderer_required_grad'] = '0'
        else:
            update_opt['renderer_required_grad'] = '1'
        train_ft_ms.train_one_scene(update_opt)

if __name__ == '__main__':
    main()
