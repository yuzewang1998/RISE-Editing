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
                            default='npbg',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='scans of checkpoints')
        parser.add_argument('--editor_checkpoints_root',
                            type=str,
                            default='/home/slam/devdata/NSEPN/comparsion',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--xyz_name',
                            type=str,
                            default='mvs_pc',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='xyz pointcloud name')

        self.opt = parser.parse_args()


def main():
    sparse = Options_npbg()
    opt = sparse.opt

    # cpc = create_checkpointscontroller(opt, 'npbg', 'lego')
    # neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    # neural_point_whole_scene.save_as_ply("lego")

    mp_trans = create_neural_point(opt,'meshlab')
    mp_trans.load_from_ply('trans')
    np_lego = create_neural_point(opt,'npbg')
    np_lego.load_from_ply('lego')
    np_trans = np_lego.select_from_meshlabpoint(mp_trans)
    np_trans.save_as_ply("trans")
    cpc = create_checkpointscontroller(opt, 'npbg', 'lego')
    cpc.set_and_save(np_trans)
if __name__=="__main__":
    # main()
    a = torch.load('/home/slam/devdata/NSEPN/comparsion/npbg/lego.pth')
    print(a)
    b = torch.load('/home/slam/devdata/NSEPN/comparsion/npbg/lego_edited.pth')
    print(b)