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


def extract_point_cloud(opt):
    cpc = create_checkpointscontroller(opt, 'penerf', 'lego')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("lego")
    cpc = create_checkpointscontroller(opt, 'penerf', 'mic')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("mic")
    cpc = create_checkpointscontroller(opt, 'penerf', 'chair')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("chair")
    cpc = create_checkpointscontroller(opt, 'penerf', 'ficus')
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("ficus")
def ficus_preprocess(opt):
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

    part = create_neural_point(opt,'penerf')
    part.load_from_ply('part')

    ficus = create_neural_point(opt, 'penerf')
    ficus.load_from_ply('ficus')

    part_t = create_neural_point(opt, 'penerf')
    part_t.load_from_ply('part_trans_90')

    ficus1 = ficus - part
    ficus2 = part_t + ficus

    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,90),np.array([-0.4,1.2,0]))
    ficus_1 = ficus1.translate(trans_matrix)
    ficus.save_as_ply('ficus1_transed')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 90), np.array([0.4, 1.2, 0]))
    ficus_2 = ficus2.translate(trans_matrix)
    ficus.save_as_ply('ficus2_transed')
    ficus = ficus_1 + ficus_2
    ficus.save_as_ply('ficus_transed')
def lego1_preprocess(opt):
    lego_nobody_mp = create_neural_point(opt,'meshlab')
    lego_nobody_mp.load_from_ply('lego1_noplane')

    lego_nobody_mp.save_as_ply('lego1')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('lego')
    tmp = origin_lego_np - lego_nobody_mp
    tmp.save_as_ply('lego1_plane')
    lego_nobody_np = origin_lego_np.select_from_meshlabpoint(lego_nobody_mp)
    lego_trans_np = lego_nobody_np.change_scale(scale_factor=0.5)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(10, -10, 60), np.array([-0.1, 0, -0.025]))
    lego_trans_np = lego_trans_np.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    lego_trans_np.save_as_ply("lego1_transed")
    # origin_lego_np = create_neural_point(opt,'penerf')
    # origin_lego_np.load_from_ply('lego')
    # lego_nobody_np = origin_lego_np
    # lego_trans_np = lego_nobody_np.change_scale(scale_factor=0.3)
    # trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-8, 0, -30), np.array([+0.025, -0.02, 0.020]))
    # lego_trans_np = lego_trans_np.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    # lego_trans_np.save_as_ply("lego1_transed")
def lego2_preprocess(opt):
    lego_nobody_mp = create_neural_point(opt,'meshlab')
    lego_nobody_mp.load_from_ply('lego1_noplane')
    origin_lego_np = create_neural_point(opt,'penerf')
    origin_lego_np.load_from_ply('lego')
    lego_nobody_np = origin_lego_np.select_from_meshlabpoint(lego_nobody_mp)
    # lego_nobody_np = create_neural_point(opt, 'penerf')
    # lego_nobody_np.load_from_ply('lego_valina')
    scraper_mp = create_neural_point(opt, 'meshlab')
    scraper_mp.load_from_ply('lego2_scraper')
    scraper_np = lego_nobody_np.select_from_meshlabpoint(scraper_mp)
    body = lego_nobody_np - scraper_np
    body.save_as_ply("lego2_body")
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(-45, 0, 0), np.array([0, 0, 0]))
    scraper_np_translated = scraper_np.translate(trans_matrix, rotate_centerpoint=np.array([0, -0.05, 0.15]))
    scraper_np_translated.save_as_ply("lego2_scraper_np_translated")
    lego_translated = body + scraper_np_translated
    lego_translated.save_as_ply("lego2_lego_translated")
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 0, 90), np.array([1.8, 0, -0.6]))
    lego_translated = lego_translated.translate(trans_matrix, rotate_centerpoint=np.array([0, 0, 0]))
    lego_translated.save_as_ply("lego2_transed")

def mic_preprocess(opt):
    mic = create_neural_point(opt,'penerf')
    mic.load_from_ply('mic')
    mic = mic.change_scale(scale_factor=0.4)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(0, 15, -45), np.array([-0.40, -0.35, 0.2]))
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
    chair.load_from_ply('chair')

    ficus = create_neural_point(opt, 'penerf')
    ficus.load_from_ply('ficus_transed')

    composite_scene = mic + lego1 + lego2 + chair + ficus

    composite_scene.save_as_ply("composite_scene")

    cpc = create_checkpointscontroller(opt, 'penerf', 'chair')
    cpc.set_and_save(composite_scene,'composite_scene')
def main():
    sparse = Options()
    opt = sparse.opt
    # extract_point_cloud(opt)
    # ficus_preprocess(opt)
    mic_preprocess(opt)
    # lego1_preprocess(opt)
    # lego2_preprocess(opt)
    composite(opt)
if __name__=="__main__":
    main()