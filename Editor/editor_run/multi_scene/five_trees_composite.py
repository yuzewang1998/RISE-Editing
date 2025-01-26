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
def editing_tree1(opt):
    # pot = create_neural_point(opt,'meshlab')
    # pot.load_from_ply('pot')
    # origin_np = create_neural_point(opt,'penerf')
    # origin_np.load_from_ply('origin')
    # pot = origin_np.select_from_meshlabpoint(pot)
    # pot.save_as_ply("pot")
    delete_mp = create_neural_point(opt,'meshlab')
    delete_mp.load_from_ply('tree1_delete')
    origin_np = create_neural_point(opt,'penerf')
    origin_np.load_from_ply('origin')
    delete_np = origin_np.select_from_meshlabpoint(delete_mp)
    tree1 = origin_np-delete_np
    tree1.save_as_ply("tree1")
    # cpc = create_checkpointscontroller(opt, 'penerf', '430000_net_ray_marching')
    # cpc.set_and_save(tree1,'tree1')
def rotate_part(opt):
    part = create_neural_point(opt,'meshlab')
    part.load_from_ply('part')
    origin_np = create_neural_point(opt,'penerf')
    origin_np.load_from_ply('ficus')
    part = origin_np.select_from_meshlabpoint(part)
    part.save_as_ply("part")
    # origin = create_neural_point('penerf')
    # origin.load_from_ply('origin')
    for i in range(11):
        deg = (i + 1) * 30
        trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, deg), np.array([0, 0, 0]))
        part_translated = part.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
        part_translated.save_as_ply('part_trans_{}'.format(deg))
def editing_tree2(opt):
    part1 = create_neural_point(opt,'penerf')
    part2 = create_neural_point(opt, 'penerf')
    part3 = create_neural_point(opt, 'penerf')
    part4 = create_neural_point(opt,'penerf')
    part5 = create_neural_point(opt, 'penerf')
    part6 = create_neural_point(opt,'penerf')
    part7 = create_neural_point(opt, 'penerf')
    part1.load_from_ply('part_trans_30')
    part2.load_from_ply('part_trans_90')
    part3.load_from_ply('part_trans_150')
    part4.load_from_ply('part_trans_210')
    part5.load_from_ply('part_trans_270')
    part6.load_from_ply('part_trans_330')
    part7.load_from_ply('pot')

    new_scene = part1 + part2 + part3 + part4 + part5 + part6 + part7
    new_scene.save_as_ply('tree2')
    cpc = create_checkpointscontroller(opt, 'penerf', '430000_net_ray_marching')
    cpc.set_and_save(new_scene,'tree2')

def editing_tree4(opt):
    # pot = create_neural_point(opt,'meshlab')
    # pot.load_from_ply('pot')
    # origin_np = create_neural_point(opt,'penerf')
    # origin_np.load_from_ply('origin')
    # pot = origin_np.select_from_meshlabpoint(pot)
    # pot.save_as_ply("pot")
    tree4 = create_neural_point(opt,'meshlab')
    tree4.load_from_ply('tree4')
    origin_np = create_neural_point(opt,'penerf')
    origin_np.load_from_ply('origin')
    tree4 = origin_np.select_from_meshlabpoint(tree4)
    tree4.save_as_ply("tree4")
    cpc = create_checkpointscontroller(opt, 'penerf', '430000_net_ray_marching')
    cpc.set_and_save(tree4,'tree4')
def composite_trees(opt):
    tree0 = create_neural_point(opt,'penerf')
    tree1 = create_neural_point(opt, 'penerf')
    tree2 = create_neural_point(opt, 'penerf')
    tree3 = create_neural_point(opt, 'penerf')
    tree4 = create_neural_point(opt,'penerf')
    tree0.load_from_ply('tree0')
    tree1.load_from_ply('tree1')
    tree2.load_from_ply('tree2')
    tree3.load_from_ply('tree3')
    tree4.load_from_ply('tree4')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([-0.6,-0.3,0]))
    tree0 = tree0.translate(trans_matrix)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([0,-0.3,0]))
    tree1 = tree1.translate(trans_matrix)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([0.6,-0.3,0]))
    tree2 = tree2.translate(trans_matrix)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([-0.25,0.3,0]))
    tree3 = tree3.translate(trans_matrix)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([0.35,0.3,0]))
    tree4 = tree4.translate(trans_matrix)
    tree0.save_as_ply("transed_tree0")
    tree1.save_as_ply('transed_tree1')
    tree2.save_as_ply('transed_tree2')
    tree3.save_as_ply('transed_tree3')
    tree4.save_as_ply('transed_tree4')
    scene = tree0+tree1+tree2+tree3+tree4
    scene.save_as_ply('final_no_rot')
    cpc = create_checkpointscontroller(opt, 'penerf', '430000_net_ray_marching')
    cpc.set_and_save(scene,'final_norot')
def main():
    sparse = Options()
    opt = sparse.opt
    # extract_neural_point(opt)
    # editing_tree4(opt)
    rotate_part(opt)
    # add_part(opt)
    # composite_trees(opt)
if __name__=="__main__":
    main()