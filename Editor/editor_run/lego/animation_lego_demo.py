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
def translate_scraper(opt,scraper_np,rot_angle = 0):

    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-rot_angle,0,0),np.array([0,0,0]))
    scraper_np_translated = scraper_np.translate(trans_matrix,rotate_centerpoint=np.array([0,-0.05,0.15]))
    return scraper_np_translated




import shutil
def main():
    sparse = Options()
    opt = sparse.opt
    iter = 500000
    meshlab_point_2_neural_point(opt)
    scraper_np = create_neural_point(opt,'penerf')
    scraper_np.load_from_ply('scraper')
    body = create_neural_point(opt, 'penerf')
    body.load_from_ply('body')
    cpc = create_checkpointscontroller(opt, 'penerf', '500000_net_ray_marching')
    for i in range(1,201):
        angle = 45/200*i
        scraper_np_translated = translate_scraper(opt,scraper_np,angle)
        lego_translated = body + scraper_np_translated
        iter += 1
        cpc.set_and_save(lego_translated, str(iter),True)
        shutil.copy(os.path.join(opt.editor_checkpoints_root,str(50000)+'_states.pth'), os.path.join(opt.editor_checkpoints_root,str(iter)+'_states.pth'))

if __name__=="__main__":
    main()