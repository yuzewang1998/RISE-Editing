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
    cpc = create_checkpointscontroller(opt, 'penerf', '950000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin")
def meshlab_point_2_neural_point(opt):
    p1_mp = create_neural_point(opt,'meshlab')
    p1_mp.load_from_ply('part1')
    origin_drums_np = create_neural_point(opt,'penerf')
    origin_drums_np.load_from_ply('origin')
    p1_np = origin_drums_np.select_from_meshlabpoint(p1_mp)
    p1_np.save_as_ply("part1")

    p2_mp = create_neural_point(opt,'meshlab')
    p2_mp.load_from_ply('part2')
    p2_np = origin_drums_np.select_from_meshlabpoint(p2_mp)
    p2_np.save_as_ply("part2")

    body = origin_drums_np - p1_np - p2_np
    body.save_as_ply("body")
def rotate_part(opt):
    part = create_neural_point(opt,'penerf')
    part.load_from_ply('part1')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 45), np.array([0,0 , 0.0]))
    part_translated = part.translate(trans_matrix, rotate_centerpoint=np.array([-0.2, 0.0, -0.1715]))
    part_translated.save_as_ply('part1_trans')

    part2 = create_neural_point(opt,'penerf')
    part2.load_from_ply('part2')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 0), np.array([0.5,0 , 0.0]))
    part_translated = part2.translate(trans_matrix)
    part_translated.save_as_ply('part2_trans')
def add_part(opt):
    part1 = create_neural_point(opt,'penerf')
    part1.load_from_ply('part1_trans')
    part2 = create_neural_point(opt,'penerf')
    part2.load_from_ply('part2_trans')
    body =  create_neural_point(opt,'penerf')
    body.load_from_ply('body')
    new_scene = body + part1 + part2
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'penerf', '950000_net_ray_marching')
    cpc.set_and_save(new_scene,'new')

def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    # meshlab_point_2_neural_point(opt)
    rotate_part(opt)
    add_part(opt)
if __name__=="__main__":
    main()