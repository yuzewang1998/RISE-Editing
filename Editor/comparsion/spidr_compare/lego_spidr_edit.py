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
def extract_neural_point(opt,neu_method,ckpt_filename):
    cpc = create_checkpointscontroller(opt, neu_method, ckpt_filename)
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_lego")
def meshlab_point_2_neural_point(opt,neu_method):
    scraper_mp = create_neural_point(opt,'meshlab')
    scraper_mp.load_from_ply('scraper')
    origin_lego_np = create_neural_point(opt,neu_method)
    origin_lego_np.load_from_ply('origin_lego')
    scraper_np = origin_lego_np.select_from_meshlabpoint(scraper_mp)
    scraper_np.save_as_ply("scraper")
    body = origin_lego_np - scraper_np
    body.save_as_ply("body")
def translate_scraper(opt,scraper_np,rot_angle = 0):

    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-rot_angle,0,0),np.array([0,0,0]))
    scraper_np_translated = scraper_np.translate(trans_matrix,rotate_centerpoint=np.array([0,-0.05,0.15]))
    return scraper_np_translated




import shutil
def main():
    sparse = Options()
    opt = sparse.opt

    opt.editor_checkpoints_root = "/home/wangyz/Documents/project/SPIDR/checkpoints/nerfsynth_sdf/lego"
    iter = 180000
    neu_method = 'spidr'


    ckpt_name = str(iter) + '_net_ray_marching'

    # os.makedirs(os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans), exist_ok=True)
    # src_path_file = os.path.join(opt.editor_checkpoints_root,ckpt_name+'.pth')
    # dst_path_file = os.path.join(os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans),ckpt_name+'.pth')
    # shutil.copy(src_path_file,dst_path_file)
    # shutil.copy(src_path_file,os.path.join(opt.editor_checkpoints_root,ckpt_name+'-o.pth'))
    # # Step1 : extract
    # extract_neural_point(opt,neu_method,ckpt_name)
    # # Step2 ： manipulate via Meshlab, save as ply,name as 'srapper_meshlabpoint.ply'
    # pass
    # Step3 :propagate
    meshlab_point_2_neural_point(opt,neu_method)
    scraper_np = create_neural_point(opt,neu_method)
    scraper_np.load_from_ply('scraper')
    body = create_neural_point(opt, neu_method)
    body.load_from_ply('body')
    cpc = create_checkpointscontroller(opt, neu_method, ckpt_name)
    angle = 45.0
    scraper_np_translated = translate_scraper(opt, scraper_np, angle)
    scraper_np_translated.save_as_ply('transd_scraper')
    lego_translated = body + scraper_np_translated
    lego_translated.save_as_ply("edited_lego")
    cpc.set_and_save(lego_translated, str(iter+1), True)
    shutil.copy(os.path.join(opt.editor_checkpoints_root, str(iter) + '_states.pth'),os.path.join(opt.editor_checkpoints_root, str(iter+1) + '_states.pth'))


if __name__=="__main__":
    main()