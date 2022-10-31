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
                            default='sc113',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
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
                            default='full',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='xyz pointcloud name')

        self.opt = parser.parse_args()

def extract(opt):
    cpc = create_checkpointscontroller(opt, 'npbg', 'scan113')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_scene")

def process(opt):
    origin_scene = create_neural_point(opt,'npbg')
    origin_scene.load_from_ply('origin_scene')
    chair = create_neural_point(opt,'meshlab')
    chair.load_from_ply('chair')
    chair = origin_scene.select_from_meshlabpoint(chair)
    chair.save_as_ply('chair')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,15),np.array([0,-0.8,0]))
    transd_chair = chair.translate(trans_matrix,rotate_itself=True)
    transd_chair.save_as_ply('transd_chair')
def editing(opt):
    origin_scene = create_neural_point(opt,'npbg')
    origin_scene.load_from_ply('origin_scene')
    transd_chair = create_neural_point(opt,'npbg')
    transd_chair.load_from_ply('transd_chair')
    new_scene = transd_chair + origin_scene
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'npbg', 'scan113')
    cpc.set_and_save(new_scene,'new')

if __name__=="__main__":
    sparse = Options_npbg()
    opt = sparse.opt
    # extract(opt)
    process(opt)
    editing((opt))
