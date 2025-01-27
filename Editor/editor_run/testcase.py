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


def tc0_cpc2mp(opt):
    cpc = create_checkpointscontroller(opt,'penerf',None)
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("a")
def tc1_mp2np(opt):
    mp_b = create_neural_point(opt,'meshlab')
    mp_b.load_from_ply('b')
    np_a = create_neural_point(opt,'penerf')
    np_a.load_from_ply('a')
    np_b = np_a.select_from_meshlabpoint(mp_b)
    np_b.save_as_ply("b")
    cpc = create_checkpointscontroller(opt, 'penerf', '250000_net_ray_marching')
    cpc.set_and_save(np_b,"b")
def tc2_sub_oprator(opt):
    a = create_neural_point(opt,'penerf')
    a.load_from_ply('a')
    b = create_neural_point(opt,'penerf')
    b.load_from_ply('b')
    c = a-b
    c.save_as_ply('a-b')
def tc3_add_oprator(opt):
    c = create_neural_point(opt,'penerf')
    c.load_from_ply('a-b')
    b = create_neural_point(opt,'penerf')
    b.load_from_ply('b')
    a = b+c
    a.save_as_ply('a_new')
def tc4_translate1(opt):
    b = create_neural_point(opt,'penerf')
    b.load_from_ply('b')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(90,0,0),np.array([5,0,0]))
    b = b.translate(trans_matrix)
    b.save_as_ply('b_trans9000500')
def tc4_translate2(opt):
    b = create_neural_point(opt,'penerf')
    b.load_from_ply('b')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(90,0,0),np.array([0,0,0]))
    b = b.translate(trans_matrix,rotate_itself=True)
    b.save_as_ply('b_trans9000000_rotateitself')
def tc4_translate3(opt):
    b = create_neural_point(opt,'penerf')
    b.load_from_ply('b')
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(90,0,0),np.array([0,0,0]))
    b = b.translate(trans_matrix,rotate_centerpoint=np.array([0,4,0]))
    b.save_as_ply('b_trans9000000_center040')
def tc5_changescale(opt):
    a = create_neural_point(opt, 'penerf')
    a.load_from_ply('a')
    a_s2 = a.change_scale(scale_factor=2.0)
    a_s2.save_as_ply('a_scale2')
    a_half = a.change_scale(scale_factor=0.5)
    a_half.save_as_ply('a_scale_half')
    a_shear = a.change_scale(scale_factor=[1,3,1])
    a_shear.save_as_ply('a_scale_shear')
def tc6_spaw_renderer(opt):
    net_paras = create_checkpointscontroller(opt, 'penerf', '905000_net_ray_marching')
    renderer_paras = create_checkpointscontroller(opt, 'penerf', 'chair')
    net_paras.aggrator_paras_copy(renderer_paras)
    net_paras.set_and_save(penerf_neuralpoint=None,edit_name=None)
def tc7_spaw_renderer(opt):
    lego_paras = create_checkpointscontroller(opt, 'penerf', '66800_net_ray_marching')
    chair_paras = create_checkpointscontroller(opt, 'penerf', 'chair')
    lego_paras.aggrator_paras_copy(chair_paras)
    lego_paras.set_and_save(penerf_neuralpoint=None,edit_name=None)
def tc8_downsample_meshlab(opt):
    mp_b = create_neural_point(opt,'meshlab')
    mp_b.load_from_ply('b')
    c = mp_b.down_sample()
    c.save_as_ply("b_downsample")

def tc9_downsample_penerfpoint(opt):
    np_b = create_neural_point(opt,'penerf')
    np_b.load_from_ply('origin_scene')
    c = np_b.down_sample()
    c.save_as_ply("origin_scene_downsample")

def tc10_mp_2_cpc(opt):
    cpc = create_checkpointscontroller(opt, 'penerf', '128700_net_ray_marching')
    np_b = create_neural_point(opt,'penerf')
    np_b.load_from_ply('origin_scene')
    np_b = np_b.change_scale(scale_factor=0.25)
    np_b = np_b.down_sample()
    cpc.set_and_save(penerf_neuralpoint=np_b, edit_name='downsample_scale0.25_downsample')
if __name__=="__main__":
    sparse = Options()
    opt = sparse.opt
    tc6_spaw_renderer(opt)
