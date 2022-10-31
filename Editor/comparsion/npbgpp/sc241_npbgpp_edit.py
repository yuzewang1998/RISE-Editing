import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from Editor.checkpoints_controller import create_checkpointscontroller
from Editor.points import create_neural_point
from utils_mine.utilize import cauc_RotationMatrix,cauc_transformationMatrix
import cv2
from tqdm import tqdm

from Editor.editor_options import Options
import numpy as np

import argparse
class Options_npbg:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Argparse of  point_editor")
        parser.add_argument('--editor_checkpoints_scans',
                            type=str,
                            default='sc241',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='scans of checkpoints')
        parser.add_argument('--editor_checkpoints_root',
                            type=str,
                            default='/home/slam/devdata/NSEPN/comparsion/npbgpp',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--xyz_name',
                            type=str,
                            default='full',#mvs_pc
                            help='xyz pointcloud name')

        self.opt = parser.parse_args()

def sc241_extract(opt):
    cpc = create_checkpointscontroller(opt, 'npbg', 'scan241')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("sc241")
def sc241_process(opt):
    origin_scene = create_neural_point(opt,'npbg')
    origin_scene.load_from_ply('sc241')
    box = create_neural_point(opt,'meshlab')
    box.load_from_ply('box')
    box = origin_scene.select_from_meshlabpoint(box)
    bkg =origin_scene - box
    bkg.save_as_ply('bkg')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([0,+0.2,0]))
    box = box.translate(trans_matrix,rotate_itself=True)
    box.save_as_ply('box')

def editing(opt):
    bkg = create_neural_point(opt,'npbg')
    bkg.load_from_ply('bkg')
    box = create_neural_point(opt,'npbg')
    box.load_from_ply('box')
    new_scene = box + bkg
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'npbg', 'scan241')
    cpc.set_and_save(new_scene, 'new_scene')
    print('?')

if __name__=="__main__":
    sparse = Options_npbg()
    opt = sparse.opt
    # sc241_extract(opt)
    sc241_process(opt)
    editing(opt)

