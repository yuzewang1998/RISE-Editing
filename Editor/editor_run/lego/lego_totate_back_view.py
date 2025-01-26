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
def meshlab_point_2_neural_point(opt):
    wl = create_neural_point(opt,'meshlab')
    wl.load_from_ply('wheel_left')
    wr = create_neural_point(opt,'meshlab')
    wr.load_from_ply('wheel_right')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('origin_lego')
    wl = origin_lego_np.select_from_meshlabpoint(wl)
    wl.save_as_ply("wheel_left")
    wr = origin_lego_np.select_from_meshlabpoint(wr)
    wr.save_as_ply("wheel_right")
    body = origin_lego_np - wr - wl
    body.save_as_ply("body")
def translate_scraper(opt):
    scraper_np = create_neural_point(opt,'penerf')
    scraper_np.load_from_ply('wheel_left')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,45,0),np.array([0,0,0]))
    scraper_np_translated = scraper_np.translate(trans_matrix,rotate_itself=True)
    scraper_np_translated.save_as_ply('test')
def add_transd_scraper_and_body(opt):
    body = create_neural_point(opt, 'penerf')
    body.load_from_ply('body')
    scraper_np_translated = create_neural_point(opt,'penerf')
    scraper_np_translated.load_from_ply('test')
    lego_translated = body + scraper_np_translated
    lego_translated.save_as_ply("transd_lego_trans[-45,0,0]_center[0,-0.05,0.15]")
    cpc = create_checkpointscontroller(opt, 'penerf', '500000_net_ray_marching')
    cpc.set_and_save(lego_translated,'transd_lego_trans[-45,0,0]_center[0,-0.05,0.15]')


if __name__ == '__main__':
    sparse = Options()
    opt = sparse.opt
    meshlab_point_2_neural_point(opt)
    translate_scraper(opt)
    # add_transd_scraper_and_body(opt)
    # extract_neural_point(opt)



