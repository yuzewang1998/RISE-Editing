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
                            default='lego_ft_edit_1',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
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
    cpc = create_checkpointscontroller(opt, 'npbg', 'lego')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("lego")



def meshlab_point_2_neural_point(opt):
    scraper_mp = create_neural_point(opt,'meshlab')
    scraper_mp.load_from_ply('scraper')
    origin_lego_np = create_neural_point(opt,'npbg')
    origin_lego_np.load_from_ply('lego')
    scraper_np = origin_lego_np.select_from_meshlabpoint(scraper_mp)
    scraper_np.save_as_ply("scraper")
    body = origin_lego_np - scraper_np
    body.save_as_ply("body")
def translate_scraper(opt):
    scraper_np = create_neural_point(opt,'npbg')
    scraper_np.load_from_ply('scraper')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-45,0,0),np.array([0,0,0]))
    scraper_np_translated = scraper_np.translate(trans_matrix,rotate_centerpoint=np.array([0,-0.05,0.15]))
    scraper_np_translated.save_as_ply('scraper_trans[-45,0,0]_center[0,-0.05,0.15]')
def add_transd_scraper_and_body(opt):
    body = create_neural_point(opt, 'npbg')
    body.load_from_ply('body')
    scraper_np_translated = create_neural_point(opt,'npbg')
    scraper_np_translated.load_from_ply('scraper_trans[-45,0,0]_center[0,-0.05,0.15]')
    lego_translated = body + scraper_np_translated
    lego_translated.save_as_ply("transd_lego_trans[-45,0,0]_center[0,-0.05,0.15]")
    cpc = create_checkpointscontroller(opt, 'npbg', 'lego')
    cpc.set_and_save(lego_translated,'transd_lego_trans[-45,0,0]_center[0,-0.05,0.15]')



def editing1(opt):
    meshlab_point_2_neural_point(opt)
    translate_scraper(opt)
    add_transd_scraper_and_body(opt)

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
    editing1(opt)
