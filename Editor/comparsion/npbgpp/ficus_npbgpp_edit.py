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
                            default='ficus_edit',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
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
                            default='mvs_pc',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='xyz pointcloud name')

        self.opt = parser.parse_args()

def extract(opt):
    cpc = create_checkpointscontroller(opt, 'npbg', 'ficus')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("ficus")

def meshlab_point_2_neural_point(opt):
    part = create_neural_point(opt,'meshlab')
    part.load_from_ply('part')
    origin_scene = create_neural_point(opt,'npbg')
    origin_scene.load_from_ply('ficus')
    scraper_np = origin_scene.select_from_meshlabpoint(part)
    scraper_np.save_as_ply("part")

def rotate_part(opt):
    part = create_neural_point(opt,'npbg')
    part.load_from_ply('part')
    # origin = create_neural_point('penerf')
    # origin.load_from_ply('origin')
    for i in range(11):
        deg = (i + 1) * 30
        trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, deg), np.array([0, 0, 0]))
        part_translated = part.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
        part_translated.save_as_ply('part_trans_{}'.format(deg))
def add_part(opt):
    part1 = create_neural_point(opt,'npbg')
    part2 = create_neural_point(opt, 'npbg')
    part3 = create_neural_point(opt, 'npbg')
    part4 = create_neural_point(opt,'npbg')
    part1.load_from_ply('part_trans_300')
    part2.load_from_ply('part_trans_60')
    part3.load_from_ply('part_trans_150')
    part4.load_from_ply('part_trans_210')
    origin =  create_neural_point(opt,'npbg')
    origin.load_from_ply('ficus')
    new_scene = origin + part1 + part2 + part3 + part4
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'npbg', 'ficus')
    cpc.set_and_save(new_scene,'new')


def editing2(opt):
    head_mp = create_neural_point(opt, 'meshlab')
    head_mp.load_from_ply('scraper_head')
    origin_lego_np = create_neural_point(opt, 'npbg')
    origin_lego_np.load_from_ply('lego')
    head_np = origin_lego_np.select_from_meshlabpoint(head_mp)
    body = origin_lego_np - head_np
    head_np = head_np.change_scale(scale_factor=[1.5,1,1.5])
    scaled_lego = head_np + body
    scaled_lego.save_as_ply("scaled_lego_[1.5,1,1.5]")
    cpc = create_checkpointscontroller(opt, 'npbg', 'lego')
    cpc.set_and_save(scaled_lego,'scaled_lego_[1.5,1,1.5]')
def editing3(opt):
    part = create_neural_point(opt, 'meshlab')
    part.load_from_ply('local_edit')
    origin_lego_np = create_neural_point(opt, 'npbg')
    origin_lego_np.load_from_ply('lego')
    part = origin_lego_np.select_from_meshlabpoint(part)
    part.save_as_ply("local_edit")
    cpc = create_checkpointscontroller(opt, 'npbg', 'lego')
    cpc.set_and_save(part,'local_edit')

if __name__=="__main__":
    sparse = Options_npbg()
    opt = sparse.opt
    # extract(opt)
    # meshlab_point_2_neural_point(opt)
    # editing1(opt)
    # rotate_part(opt)
    add_part(opt)