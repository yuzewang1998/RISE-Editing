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

def extract(opt):
    cpc = create_checkpointscontroller(opt, 'penerf', '765000_net_ray_marching')
    origin_scene = cpc.cvt_2_neuralPoint()
    origin_scene.save_as_ply("origin_scene")
def process(opt):
    origin_scene = create_neural_point(opt,'penerf')
    origin_scene.load_from_ply('origin_scene')
    chair = create_neural_point(opt,'meshlab')
    chair.load_from_ply('chair')
    chair = origin_scene.select_from_meshlabpoint(chair)
    chair.save_as_ply('chair')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,15),np.array([0,-0.8,0]))
    transd_chair = chair.translate(trans_matrix,rotate_itself=True)
    transd_chair.save_as_ply('transd_chair')
def editing(opt):
    origin_scene = create_neural_point(opt,'penerf')
    origin_scene.load_from_ply('origin_scene')
    transd_chair = create_neural_point(opt,'penerf')
    transd_chair.load_from_ply('transd_chair')
    new_scene = transd_chair + origin_scene
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'penerf', '765000_net_ray_marching')
    cpc.set_and_save(new_scene, 'new_scene')
    print('?')
def main():
    sparse = Options()
    opt = sparse.opt
    editing(opt)

if __name__=="__main__":
    main()