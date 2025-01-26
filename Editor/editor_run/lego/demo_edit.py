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
def extract_neural_point(opt):
    cpc = create_checkpointscontroller(opt, 'penerf', '500000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_lego")
def meshlab_point_2_neural_point(opt):
    scraper_mp = create_neural_point(opt,'meshlab')
    scraper_mp.load_from_ply('scraper')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('origin_lego')
    scraper_np = origin_lego_np.select_from_meshlabpoint(scraper_mp)
    scraper_np.save_as_ply("scraper")
    body = origin_lego_np - scraper_np
    body.save_as_ply("body")
def translate_scraper(opt):
    scraper_np = create_neural_point(opt,'penerf')
    scraper_np.load_from_ply('scraper')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-45,0,0),np.array([0,0,0]))
    scraper_np_translated = scraper_np.translate(trans_matrix,rotate_centerpoint=np.array([0,-0.05,0.15]))
    scraper_np_translated.save_as_ply('scraper_trans[-45,0,0]_center[0,-0.05,0.15]')
def add_transd_scraper_and_body(opt):
    body = create_neural_point(opt, 'penerf')
    body.load_from_ply('body')
    scraper_np_translated = create_neural_point(opt,'penerf')
    scraper_np_translated.load_from_ply('scraper_trans[-45,0,0]_center[0,-0.05,0.15]')
    lego_translated = body + scraper_np_translated
    lego_translated.save_as_ply("transd_lego_trans[-45,0,0]_center[0,-0.05,0.15]")
    cpc = create_checkpointscontroller(opt, 'penerf', '500000_net_ray_marching')
    cpc.set_and_save(lego_translated,'transd_lego_trans[-45,0,0]_center[0,-0.05,0.15]')



def edit1_transed_scraper(opt):
    meshlab_point_2_neural_point(opt)
    translate_scraper(opt)
    add_transd_scraper_and_body(opt)
def edit2_scaleup_scraper(opt):
    head_mp = create_neural_point(opt, 'meshlab')
    head_mp.load_from_ply('scraper_head')
    origin_lego_np = create_neural_point(opt, 'penerf')
    origin_lego_np.load_from_ply('origin_lego')
    trash = create_neural_point(opt, 'meshlab')
    trash.load_from_ply('lego_fix_trash')
    trash_np = origin_lego_np.select_from_meshlabpoint(trash)
    head_np = origin_lego_np.select_from_meshlabpoint(head_mp)
    origin_lego_np_fix = origin_lego_np - trash_np
    origin_lego_np_fix.save_as_ply('fl')
    body = origin_lego_np_fix - head_np
    head_np = head_np.change_scale(scale_factor=[1.4,1.4,1.4])
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([0,-0.05,-0.05]))
    head_np = head_np.translate(trans_matrix,rotate_centerpoint=np.array([0,0,0.0]))
    head_np.save_as_ply("scaled_head_[1.5,1,1.5]")
    scaled_lego = head_np + body
    scaled_lego.save_as_ply("scaled_lego_[1.5,1,1.5]")
    cpc = create_checkpointscontroller(opt, 'penerf', '500000_net_ray_marching-o')
    cpc.set_and_save(scaled_lego,'scaled_lego_[1.5,1,1.5]')
def edit3_localdelete_scraper(opt):
    part = create_neural_point(opt, 'meshlab')
    part.load_from_ply('local_edit')
    origin_lego_np = create_neural_point(opt, 'penerf')
    origin_lego_np.load_from_ply('origin_lego')
    part = origin_lego_np.select_from_meshlabpoint(part)
    part.save_as_ply("part")
    cpc = create_checkpointscontroller(opt, 'penerf', 'lego')
    cpc.set_and_save(part,'local_edit')
def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    edit1_transed_scraper(opt)
    # edit2_scaleup_scraper(opt)
    # edit3_scaleup_scraper(opt)
if __name__=="__main__":
    main()