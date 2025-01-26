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
    cpc = create_checkpointscontroller(opt, 'penerf', '900000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin")
def meshlab_point_2_neural_point(opt):
    part = create_neural_point(opt,'meshlab')
    part.load_from_ply('part')
    origin = create_neural_point(opt,'penerf')
    origin.load_from_ply('origin')
    part = origin.select_from_meshlabpoint(part)
    part.save_as_ply("part")
    sit = origin - part
    sit.save_as_ply('sit')
def rotate_part(opt):
    part = create_neural_point(opt,'penerf')
    part.load_from_ply('part')
    part = part.change_scale(scale_factor=[1,1,1.5])
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-30, 0, 0), np.array([0, 0.18, -0.045]))
    part_translated = part.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    part_translated.save_as_ply('part_trans_{}'.format(30))
def add_part(opt):
    part1 = create_neural_point(opt,'penerf')
    part2 = create_neural_point(opt, 'penerf')
    part1.load_from_ply('part_trans_30')
    part2.load_from_ply('sit')
    new_scene = part1 + part2
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'penerf', '900000_net_ray_marching')
    cpc.set_and_save(new_scene,'new')


def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    # meshlab_point_2_neural_point(opt)
    # rotate_part(opt)
    add_part(opt)
if __name__=="__main__":
    main()