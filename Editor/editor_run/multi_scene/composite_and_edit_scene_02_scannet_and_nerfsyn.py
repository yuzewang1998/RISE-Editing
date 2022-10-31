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
    cpc = create_checkpointscontroller(opt, 'penerf', '590000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("sc113")
    cpc = create_checkpointscontroller(opt, 'penerf', 'nerf_chair')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("nerf_chair")
def preprocess(opt):

    nerf_chair = create_neural_point(opt,'penerf')
    nerf_chair.load_from_ply('nerf_chair')
    nerf_chair = nerf_chair.change_scale(scale_factor=0.5)

    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,90),np.array([3.4,2.2,0.55]))
    nerf_chair = nerf_chair.translate(trans_matrix)
    nerf_chair.save_as_ply("nerf_chair_trans")
def editing(opt):
    sc113_chair = create_neural_point(opt,'meshlab')
    sc113_chair.load_from_ply('sc113_chair')
    origin_np = create_neural_point(opt,'penerf')
    origin_np.load_from_ply('sc113')
    sc113_chair = origin_np.select_from_meshlabpoint(sc113_chair)
    bkg = origin_np - sc113_chair
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 30), np.array([0,0, 0]))
    sc113_chair = sc113_chair.translate(trans_matrix,rotate_itself=True)
    sc113_chair.save_as_ply('sc113_trans_chair')
    bkg.save_as_ply('bkg')
    nerf_chair = create_neural_point(opt,'penerf')
    nerf_chair.load_from_ply('nerf_chair')
    nerf_chair = nerf_chair.change_scale(scale_factor=0.5)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,130),np.array([2.7,1.0,0.51]))
    nerf_chair = nerf_chair.translate(trans_matrix)
    nerf_chair.save_as_ply("nerf_chair_trans")
    result = bkg + nerf_chair +sc113_chair
    result.save_as_ply('result')
    cpc = create_checkpointscontroller(opt, 'penerf', '590000_net_ray_marching')
    cpc.set_and_save(result,'alter_chair')

def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    editing(opt)
if __name__=="__main__":
    main()