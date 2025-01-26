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



def main():
    sparse = Options()
    opt = sparse.opt
    cpc = create_checkpointscontroller(opt, 'penerf', '1000000_net_ray_marching')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_lego")
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 180), np.array([0, 0, 0]))
    neural_point_whole_scene = neural_point_whole_scene.translate(trans_matrix,rotate_itself=True)
    cpc.set_and_save(neural_point_whole_scene, '180')
if __name__=="__main__":
    main()