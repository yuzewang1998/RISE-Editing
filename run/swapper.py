
import sys
import os
import pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from Editor.checkpoints_controller import create_checkpointscontroller
from options import TrainOptions

from run import train_ft_ms

import sys
import os
import pathlib

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
        'vscale':[3,3,3],
        'zero_one_loss_items': [],
        'zero_one_loss_weights': [1.0],
        'train_step':100,
        'prune_iter':30001,
        'prune_max_iter': 200000,
        "cpc_opt": {'editor_checkpoints_root': os.path.join(opt.checkpoints_dir, opt.name.split('/')[0]),
                    'editor_checkpoints_scans': 'lego'}
    }
    scannet_nerf_chair = {
        "data_root" : "../data_src/nerf/nerf_synthetic_colmap/",
        'scan' : "chair",
        'dataset_name' : 'nerf_synth360_ft',
        'name': 'train_multi_scene/chair',
        'P': 12,
        'SR': 24,
        'default_conf': 0.15,
        'depth_occ': 1,
        'random_sample_size': 60,
        'vsize': [0.004, 0.004, 0.004],
        'vscale': [3, 3, 3],
        'zero_one_loss_items': [],
        'zero_one_loss_weights': [1.0],
        'train_step':100,
        'prune_iter': -1,
        'prune_max_iter': 200000,
        "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
                   'editor_checkpoints_scans':'chair'}

    }
    scannet_nerf_mic = {
        "data_root" : "../data_src/nerf/nerf_synthetic_colmap/",
        'scan' : "mic",
        'dataset_name' : 'nerf_synth360_ft',
        'name': 'train_multi_scene/mic',
        'P': 12,
        'SR': 80,
        'default_conf': 0.15,
        'depth_occ': 1,
        'random_sample_size': 60,
        'vsize': [0.004, 0.004, 0.004],
        'vscale': [2, 2, 2],
        'zero_one_loss_items': [],
        'zero_one_loss_weights': [1.0],
        'train_step':100,
        'prune_iter': -1,
        'prune_max_iter': 0,
        "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
                   'editor_checkpoints_scans':'mic'}

    }
    scannet_nerf_ficus = {
        "data_root" : "../data_src/nerf/nerf_synthetic_colmap/",
        'scan' : "ficus",
        'dataset_name' : 'nerf_synth360_ft',
        'name': 'train_multi_scene/ficus',
        'P': 12,
        'SR': 80,
        'default_conf': 0.15,
        'depth_occ': 1,
        'random_sample_size': 60,
        'vsize': [0.004, 0.004, 0.004],
        'vscale': [3, 3, 3],
        'zero_one_loss_items': [],
        'zero_one_loss_weights': [1.0],
        'train_step':100,
        'prune_iter': 30000,
        'prune_max_iter': 100000,
        "cpc_opt":{'editor_checkpoints_root': os.path.join(opt.checkpoints_dir,opt.name.split('/')[0]),
                   'editor_checkpoints_scans':'ficus'}

    }
    scene_list = {
        0: scannet_nerf_lego,
        1: scannet_nerf_chair,
        2: scannet_nerf_mic,
        3: scannet_nerf_ficus,

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
