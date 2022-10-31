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
    cpc = create_checkpointscontroller(opt, 'penerf', '1000000_net_ray_marching')
    origin_scene = cpc.cvt_2_neuralPoint()
    origin_scene.save_as_ply("origin_scene")
def process(opt):
    origin_scene = create_neural_point(opt,'penerf')
    origin_scene.load_from_ply('origin_scene')
    box = create_neural_point(opt,'meshlab')
    box.load_from_ply('box')
    box = origin_scene.select_from_meshlabpoint(box)
    bkg =origin_scene - box
    bkg.save_as_ply('bkg')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,0),np.array([0,+0.2,0]))
    box = box.translate(trans_matrix,rotate_itself=True)
    box.save_as_ply('box')

def editing(opt):
    bkg = create_neural_point(opt,'penerf')
    bkg.load_from_ply('bkg')
    box = create_neural_point(opt,'penerf')
    box.load_from_ply('box')
    new_scene = box + bkg
    new_scene.save_as_ply('new_scene')
    cpc = create_checkpointscontroller(opt, 'penerf', '1000000_net_ray_marching')
    cpc.set_and_save(new_scene, 'new_scene')
    print('?')
def main():
    sparse = Options()
    opt = sparse.opt
    # process(opt)
    editing(opt)

if __name__=="__main__":
    main()