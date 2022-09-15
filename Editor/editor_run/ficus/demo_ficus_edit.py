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
    cpc = create_checkpointscontroller(opt, 'penerf', '750000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_ficus")
def meshlab_point_2_neural_point(opt):
    scraper_mp = create_neural_point(opt,'meshlab')
    scraper_mp.load_from_ply('fix_ficus')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('origin_ficus')
    scraper_np = origin_lego_np.select_from_meshlabpoint(scraper_mp)
    scraper_np.save_as_ply("fix_ficus")
    cpc = create_checkpointscontroller(opt, 'penerf', '750000_net_ray_marching')
    cpc.set_and_save(scraper_np,'fix')


def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    meshlab_point_2_neural_point(opt)
if __name__=="__main__":
    main()