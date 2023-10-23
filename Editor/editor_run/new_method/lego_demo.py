import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from Editor.checkpoints_controller import create_checkpointscontroller
from Editor.points import create_neural_point
from utils_mine.utilize import cauc_RotationMatrix,cauc_transformationMatrix
import cv2
from tqdm import tqdm
import open3d as o3d
import shutil
from Editor.editor_options import Options
import numpy as np
def extract_neural_point(opt,ckpt_name):
    cpc = create_checkpointscontroller(opt, 'penerf', ckpt_name)
    neural_point_whole_scene = cpc.cvt_2_neuralPoint()
    neural_point_whole_scene.save_as_ply("origin_lego")

def rotate_pcd(pcd, translation_matrix: np.ndarray = None, rotate_itself=False,rotate_centerpoint=np.array([0, 0, 0])):
    assert translation_matrix.shape == (4, 4) and type(
        translation_matrix) == np.ndarray, 'Please input a translation matrix of 4*4'
    if rotate_itself == True:
        rotate_centerpoint = np.sum(np.asarray(pcd.point['positions']),axis=0).astype(np.float32)/np.asarray(pcd.point['positions']).shape[0]
    rot_matrix = translation_matrix[:3, :3].astype(np.float32)
    trans_vector = translation_matrix[:3, 3].astype(np.float32)
    pcd.point['positions'] = ((pcd.point['positions'].numpy().astype(np.float32) - rotate_centerpoint.astype(np.float32)) @ rot_matrix + trans_vector + rotate_centerpoint.astype(np.float32))

    dirx = np.concatenate((pcd.point['dirx0'].numpy(),pcd.point['dirx1'].numpy(),pcd.point['dirx2'].numpy()),axis=-1) @ rot_matrix
    diry = np.concatenate((pcd.point['diry0'].numpy(), pcd.point['diry1'].numpy(), pcd.point['diry2'].numpy()), axis=-1) @ rot_matrix
    dirz = np.concatenate((pcd.point['dirz0'].numpy(), pcd.point['dirz1'].numpy(), pcd.point['dirz2'].numpy()), axis=-1) @ rot_matrix
    pcd.point['dirx0'], pcd.point['dirx1'], pcd.point['dirx2'] = dirx[...,0][:,None].astype(np.float32), dirx[...,1][:,None].astype(np.float32), dirx[...,2][:,None].astype(np.float32)
    pcd.point['diry0'], pcd.point['diry1'], pcd.point['diry2'] = diry[..., 0][:,None].astype(np.float32), diry[..., 1][:,None].astype(np.float32), diry[..., 2][:,None].astype(np.float32)
    pcd.point['dirz0'], pcd.point['dirz1'], pcd.point['dirz2'] = dirz[..., 0][:,None].astype(np.float32), dirz[..., 1][:,None].astype(np.float32), dirz[..., 2][:,None].astype(np.float32)
    return pcd
def manipulate_neural_point(opt):
    dir_name = os.path.join(opt.editor_checkpoints_root, opt.editor_checkpoints_scans)
    pcd = o3d.t.io.read_point_cloud(os.path.join(dir_name, "origin_lego_penerfneuralpoint_mask.ply"))
    pcd_sel_mask = pcd.point.flags.numpy()[..., 0] > 0
    pcd_rot = pcd.select_by_mask(pcd_sel_mask)
    pcd_fix = pcd.select_by_mask(pcd_sel_mask, True)
    trans_matrix = cauc_transformationMatrix(cauc_RotationMatrix(45, 0, 0),np.array([0,0,0]))
    pcd_rot = rotate_pcd(pcd_rot,trans_matrix,rotate_itself=False,rotate_centerpoint= np.array([0, -0.7, 0.8]))

    pcd_merge = pcd_rot + pcd_fix

    dir_name = os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans)
    o3d.t.io.write_point_cloud(os.path.join(dir_name, "edited_lego_penerfneuralpoint.ply"), pcd_merge)

def back_to_ckpt(opt,ckpt_name):
    cpc = create_checkpointscontroller(opt, 'penerf', ckpt_name)
    edited_lego_np = create_neural_point(opt,'penerf')
    edited_lego_np.load_from_ply('edited_lego')
    cpc.set_and_save(edited_lego_np,'edited')
def pth_copy_paste(opt,ckpt_name):
    origin_ckpt_path = os.path.join(opt.editor_checkpoints_root,ckpt_name+".pth")
    edited_ckpt_path = os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans,ckpt_name+"_edited.pth")
    os.rename(origin_ckpt_path,os.path.join(opt.editor_checkpoints_root,ckpt_name+"-o.pth"))
    shutil.copy(edited_ckpt_path,origin_ckpt_path)
def main():
    sparse = Options()
    opt = sparse.opt
    ckpt_name = '125000_net_ray_marching'
    # Step 1 : ckpt 2 ply
    # os.mkdir(os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans))
    # shutil.copy(os.path.join(opt.editor_checkpoints_root,ckpt_name+".pth"),os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans,ckpt_name+".pth"))
    # extract_neural_point(opt,ckpt_name)


    # Step 2: select ply via SAM, app.py : A ply file will be saved with mask(flag) named xxx_mask.ply, manipulate it!
    pass


    # Step 3: manipulate ply via XX
    manipulate_neural_point(opt)


    # Step 4: back to ckpt
    back_to_ckpt(opt,ckpt_name)
    pth_copy_paste(opt,ckpt_name)

if __name__=="__main__":
    main()