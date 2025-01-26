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
    cpc = create_checkpointscontroller(opt, 'penerf', '430000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin")
def meshlab_point_2_neural_point(opt):
    scraper_mp = create_neural_point(opt,'meshlab')
    scraper_mp.load_from_ply('part')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('origin')
    scraper_np = origin_lego_np.select_from_meshlabpoint(scraper_mp)
    scraper_np.save_as_ply("part")
def rotate_part(opt):
    part = create_neural_point(opt,'penerf')
    part.load_from_ply('part')
    # origin = create_neural_point('penerf')
    # origin.load_from_ply('origin')
    for i in range(11):
        deg = (i + 1) * 30
        trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, deg), np.array([0, 0, 0]))
        part_translated = part.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
        part_translated.save_as_ply('part_trans_{}'.format(deg))
def add_part(opt):
    part1 = create_neural_point(opt,'penerf')
    part2 = create_neural_point(opt, 'penerf')
    part3 = create_neural_point(opt, 'penerf')
    part4 = create_neural_point(opt,'penerf')
    part1.load_from_ply('part_trans_300')
    part2.load_from_ply('part_trans_60')
    part3.load_from_ply('part_trans_150')
    part4.load_from_ply('part_trans_210')
    origin =  create_neural_point(opt,'penerf')
    origin.load_from_ply('origin')
    new_scene = origin + part1 + part2 + part3 + part4
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'penerf', '430000_net_ray_marching')
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