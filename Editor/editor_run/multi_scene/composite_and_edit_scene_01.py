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
def ficus_preprocess(opt):
    ficus = create_neural_point(opt,'penerf')
    ficus.load_from_ply('ficus_valina')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,90),np.array([-0.4,1.2,0]))
    ficus_1 = ficus.translate(trans_matrix)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 90), np.array([0.4, 1.2, 0]))
    ficus_2 = ficus.translate(trans_matrix)
    ficus = ficus_1 + ficus_2
    ficus.save_as_ply('ficus_transed')
def lego1_preprocess(opt):
    # lego_nobody_mp = create_neural_point(opt,'meshlab')
    # lego_nobody_mp.load_from_ply('lego1_noplane')
    # origin_lego_np = create_neural_point(opt,'penerf')
    # origin_lego_np.load_from_ply('lego_valina')
    # lego_nobody_np = origin_lego_np.select_from_meshlabpoint(lego_nobody_mp)
    # lego_trans_np = lego_nobody_np.change_scale(scale_factor=0.5)
    # trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(10, -10, 60), np.array([-0.1, 0, -0.025]))
    # lego_trans_np = lego_trans_np.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    # lego_trans_np.save_as_ply("lego1_transed")
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('lego_valina')
    lego_nobody_np = origin_lego_np
    lego_trans_np = lego_nobody_np.change_scale(scale_factor=0.3)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-8, 0, -30), np.array([+0.025, -0.02, 0.020]))
    lego_trans_np = lego_trans_np.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    lego_trans_np.save_as_ply("lego1_transed")
def lego2_preprocess(opt):
    lego_nobody_mp = create_neural_point(opt,'meshlab')
    lego_nobody_mp.load_from_ply('lego1_noplane')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('lego_valina')
    lego_nobody_np = origin_lego_np.select_from_meshlabpoint(lego_nobody_mp)
    # lego_nobody_np = create_neural_point(opt, 'penerf')
    # lego_nobody_np.load_from_ply('lego_valina')
    scraper_mp = create_neural_point(opt, 'meshlab')
    scraper_mp.load_from_ply('lego2_scraper')
    scraper_np = lego_nobody_np.select_from_meshlabpoint(scraper_mp)
    body = lego_nobody_np - scraper_np
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-45, 0, 0), np.array([0, 0, 0]))
    scraper_np_translated = scraper_np.translate(trans_matrix, rotate_centerpoint=np.array([0, -0.05, 0.15]))
    lego_translated = body + scraper_np_translated
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 90), np.array([1.8, 0, -0.6]))
    lego_translated = lego_translated.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    lego_translated.save_as_ply("lego2_transed")

def mic_preprocess(opt):
    mic = create_neural_point(opt,'penerf')
    mic.load_from_ply('mic_valina')
    mic = mic.change_scale(scale_factor=0.25)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-10, 10, -45), np.array([-0.25, -0.45, 0.08]))
    mic = mic.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    mic.save_as_ply("mic_transed")
def composite(opt):
    mic = create_neural_point(opt,'penerf')
    mic.load_from_ply('mic_transed')

    lego1 = create_neural_point(opt,'penerf')
    lego1.load_from_ply('lego1_transed')

    lego2 = create_neural_point(opt,'penerf')
    lego2.load_from_ply('lego2_transed')

    chair = create_neural_point(opt,'penerf')
    chair.load_from_ply('chair_valina')

    ficus = create_neural_point(opt, 'penerf')
    ficus.load_from_ply('ficus_transed')

    composite_scene = mic + lego1 + lego2 + chair + ficus

    composite_scene.save_as_ply("composite_scene")

    cpc = create_checkpointscontroller(opt, 'penerf', 'chair')
    cpc.set_and_save(composite_scene,'composite_scene')
def main():
    sparse = Options()
    opt = sparse.opt
    # ficus_preprocess(opt)
    # mic_preprocess(opt)
    # lego1_preprocess(opt)
    # lego2_preprocess(opt)
    composite(opt)
if __name__=="__main__":
    main()