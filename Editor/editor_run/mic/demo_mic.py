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
    cpc = create_checkpointscontroller(opt, 'penerf', '670000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin")
def meshlab_point_2_neural_point(opt):
    p_mp = create_neural_point(opt,'meshlab')
    p_mp.load_from_ply('part')
    origin_mic_np = create_neural_point(opt,'penerf')
    origin_mic_np.load_from_ply('origin')
    p_np = origin_mic_np.select_from_meshlabpoint(p_mp)
    p_np.save_as_ply("part")
    body = origin_mic_np - p_np
    body.save_as_ply("body")
def rotate_part(opt):
    part = create_neural_point(opt,'penerf')
    part.load_from_ply('part')
    # origin = create_neural_point('penerf')
    # origin.load_from_ply('origin')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-20, 0, 0), np.array([0,0 , 0.0]))
    part_translated = part.translate(trans_matrix, rotate_centerpoint=np.array([0.08, 0.10, 0.25]))
    part_translated.save_as_ply('part_trans')

def add_part(opt):
    part = create_neural_point(opt,'penerf')

    part.load_from_ply('part_trans')
    body =  create_neural_point(opt,'penerf')
    body.load_from_ply('body')
    new_scene = body + part
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'penerf', '670000_net_ray_marching')
    cpc.set_and_save(new_scene,'new')


def main():
    sparse = Options()
    opt = sparse.opt
    extract_neural_point(opt)
    meshlab_point_2_neural_point(opt)
    rotate_part(opt)
    add_part(opt)
if __name__=="__main__":
    main()