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
    cpc = create_checkpointscontroller(opt, 'penerf', 'lego')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_lego")
    cpc = create_checkpointscontroller(opt, 'penerf', 'chair')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_chair")
    # cpc = create_checkpointscontroller(opt, 'penerf', 'mic')
    # neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    # neural_point_whole_scene.save_as_ply("origin_mic")
    # cpc = create_checkpointscontroller(opt, 'penerf', 'hotdog')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_hotdog")
def meshlab_point_2_neural_point_fail_case(opt):
    lego_nobody_mp = create_neural_point(opt,'meshlab')
    lego_nobody_mp.load_from_ply('lego_noplane')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('origin_lego')
    lego_nobody_np = origin_lego_np.select_from_meshlabpoint(lego_nobody_mp)
    lego_trans_np = lego_nobody_np.change_scale(scale_factor=0.5)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(10, -10, 60), np.array([-0.1, 0, -0.025]))
    lego_trans_np = lego_trans_np.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    lego_trans_np.save_as_ply("lego_transed")
    chair_np = create_neural_point(opt,'penerf')
    chair_np.load_from_ply('origin_chair')
    comp_scene = chair_np + lego_trans_np
    comp_scene.save_as_ply("composite_[lego_chair]")
    cpc = create_checkpointscontroller(opt, 'penerf', 'renderer')
    cpc.set_and_save(comp_scene,'composite_[lego_chair]')
def meshlab_point_2_neural_point(opt):
    lego_nobody_mp = create_neural_point(opt,'meshlab')
    lego_nobody_mp.load_from_ply('lego_noplane')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('origin_lego')
    lego_nobody_np = origin_lego_np.select_from_meshlabpoint(lego_nobody_mp)
    #lego_trans_np = lego_nobody_np.change_scale(scale_factor=0.5)
    lego_trans_np = lego_nobody_np
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(10, -10, 60), np.array([-0.1, 0, -0.025]))
    lego_trans_np = lego_trans_np.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    lego_trans_np.save_as_ply("lego_transed")
    chair_np = create_neural_point(opt,'penerf')
    chair_np.load_from_ply('origin_chair')
    comp_scene = chair_np + lego_trans_np
    comp_scene.save_as_ply("composite_[lego_chair]")
    cpc = create_checkpointscontroller(opt, 'penerf', 'renderer')
    cpc.set_and_save(comp_scene,'composite_[lego_chair]')

def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    meshlab_point_2_neural_point_fail_case(opt)
if __name__=="__main__":
    main()