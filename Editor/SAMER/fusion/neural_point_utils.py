import open3d as o3d
import numpy as np
import torch


import torch.nn.functional as F
def read_neural_point(which_neuralpoint,pcd):
    if which_neuralpoint == 'ripnerf':
        n_ptr = pcd.point['positions'].shape[0] # number of ptr
        conf = pcd.point.conf.numpy() # [n,1]
        positions = pcd.point['positions'].numpy() # [n,3]
        colors = pcd.point['colors'].numpy()
        directions = {'dirx':None,'diry':None,'dirz':None}
        for ax in ['x','y','z']:
            var_vector = np.zeros([n_ptr,0])
            vec_name = 'dir'+ax
            for idx in ['0','1','2']:
                var_name = vec_name+idx
                var = pcd.point[var_name].numpy()
                var_vector = np.concatenate([var_vector,var],axis=-1)
            directions[vec_name] = var_vector
        embeddings = np.zeros([n_ptr,0])
        for i in range(32):
            var_name = 'embeding'+ str(i)
            var = pcd.point[var_name].numpy()
            embeddings = np.concatenate([embeddings,var],axis=-1)
        return {
            'conf': conf,
            'positions' : positions,
            'colors' : colors,
            'directions' : directions,
            'embeddings' : embeddings
        }
    elif which_neuralpoint == 'standard':
        return None
    else:
        raise NotImplementedError
def filter_neural_pcd(mask,pnt_neural_attr,interested_idx_list=[],cos_magic_number = 0.1):
    # mask [222812,1]
    # read from dict
    mask = mask[:,0] # [222812,]
    positions = pnt_neural_attr['positions'] #[222812,3]
    colors = pnt_neural_attr['colors'] #[222812,3]
    directionx = pnt_neural_attr['directions']['dirx']
    directiony = pnt_neural_attr['directions']['diry']
    directionz = pnt_neural_attr['directions']['dirz']
    embeddings = pnt_neural_attr['embeddings'] #[222812,32]

    # cauculate interested region
    interested_pcd_idx, interested_idx_list, interested_occ = find_interest_p2v(positions,mask,interested_idx_list)
    print('interested pcd',len(interested_pcd_idx),interested_idx_list.shape[0])

    # cauculate Similarity with neural embedding
    # interesting region
    positions_in = positions[interested_pcd_idx]
    # colors_in = colors[interested_pcd_idx]
    # directionx_in = directionx[interested_pcd_idx]
    # directiony_in = directiony[interested_pcd_idx]
    # directionz_in = directionz[interested_pcd_idx]
    embeddings_in = embeddings[interested_pcd_idx] # [n,32]
    # masked region
    positions_msk = positions[mask]
    embeddings_msk = embeddings[mask]
    embeddings_msk_avg = np.average(embeddings_msk,axis=0,keepdims=True)#cauculate the average embeddings for each channel [1,32]
    score = F.cosine_similarity(torch.from_numpy(embeddings_in), torch.from_numpy(embeddings_msk_avg))
    #score = F.normalize(F.cosine_similarity(torch.from_numpy(embeddings_in), torch.from_numpy(embeddings_msk_avg)), p=1, dim=0)[..., None] # [n,1]


    select_mask = (score >  cos_magic_number)
    print("add number of `points` for region growing:",len(select_mask[select_mask==True]))
    mask[interested_pcd_idx] = select_mask
    mask_grown_region = torch.zeros_like(torch.tensor(mask)).bool()
    mask_grown_region[interested_pcd_idx[select_mask]] = True

    return mask[...,None], len(select_mask[select_mask==True]), interested_idx_list,mask_grown_region

def filter_decline_pcd(msk,pnt_neural_attr,interested_idx_list = [],cos_magic_number = -0.4):
    # read from dict
    mask = msk[:, 0]  # [222812,]
    positions = pnt_neural_attr['positions']  # [222812,3]
    colors = pnt_neural_attr['colors']  # [222812,3]
    directionx = pnt_neural_attr['directions']['dirx']
    directiony = pnt_neural_attr['directions']['diry']
    directionz = pnt_neural_attr['directions']['dirz']
    embeddings = pnt_neural_attr['embeddings']  # [222812,32]
    positions_msk = positions[mask]
    embeddings_msk = embeddings[mask]
    embeddings_msk_avg = np.average(embeddings_msk, axis=0, keepdims=True)
    score = F.cosine_similarity(torch.from_numpy(embeddings_msk), torch.from_numpy(embeddings_msk_avg))
    select_mask = (score > cos_magic_number)
    num_of_decline = len(select_mask) - np.count_nonzero(select_mask)
    print('decline {} ptr'.format(num_of_decline))
    tag = (mask==True)
    mask[tag] = select_mask
    null = None
    mask_decline = torch.zeros_like(torch.tensor(mask)).bool()
    mask_decline[tag] = ~select_mask
    return mask, num_of_decline, null,  mask_decline
def find_interest_p2v(positions,mask,interested_idx_list = [], voxel_size = np.array([0.008,0.008,0.008]),max_ptr_in_a_occ = 64):
    lower, upper = np.min(positions, axis=0) - voxel_size, np.max(positions, axis=0) + voxel_size
    occ_shift = lower - 1e-9
    shift_positions = positions - occ_shift
    shift_coord = np.floor(shift_positions/voxel_size).astype(int)
    occ_shape = np.ceil((upper - lower) / voxel_size).astype(int)
    occ_index = np.zeros([occ_shape[0], occ_shape[1], occ_shape[2], max_ptr_in_a_occ], dtype=np.int32)  # e.g[60,50,40,32]
    occ_valid = np.zeros(occ_shape, dtype=np.int32) # occ 内有多少个点
    occ_mask = np.zeros(occ_shape, dtype=np.int32)  #occ 内有多少个mask的点
    interest_points = []
    # build occ
    for i in range(len(shift_positions)):
        coord_x,coord_y,coord_z = shift_coord[i]#[3,]
        occ_index[coord_x,coord_y,coord_z,occ_valid[coord_x,coord_y,coord_z]] = i
        occ_valid[coord_x,coord_y,coord_z] += 1
        if mask[i]:
            occ_mask[coord_x,coord_y,coord_z] += 1
    # find search region_occ
    if interested_idx_list == [] :
        interested_occ = np.zeros_like(occ_valid)
        interested_pcd = []
        # interested_idx_list = []
        for i in range(occ_shape[0]):
            for j in range(occ_shape[1]):
                for k in range(occ_shape[2]):
                    if i > 0 and i < occ_shape[0]-1 and j > 0 and j < occ_shape[1]-1 and k > 0 and k < occ_shape[2]-1:
                        # 2 types of occ is selected into occ_mask: 1.边缘：自身是1;上下左右存在0；2：边缘：自身是0上下左右存在1
                        flag1 = (occ_valid[i,j,k] > 0 ) and (occ_mask[i, j, k] == 0) and (
                                    (occ_mask[i + 1, j, k] > 0) or (occ_mask[i, j + 1, k] > 0) or (
                                        occ_mask[i, j, k + 1] > 0 ) or occ_mask[i - 1, j, k] > 0 or occ_mask[
                                        i, j - 1, k] > 0 or (occ_mask[i, j, k - 1] > 0))
                        flag2 = (occ_valid[i,j,k] > 0 ) and (occ_mask[i, j, k] == 1) and (
                                    (occ_mask[i + 1, j, k] == 0) or (occ_mask[i, j + 1, k] == 0) or (
                                        occ_mask[i, j, k + 1] == 0) or occ_mask[i - 1, j, k] == 0 or occ_mask[
                                        i, j - 1, k] == 0 or (occ_mask[i, j, k - 1] == 0))
                        if flag1 or flag2:# all point is interesting
                            interested_occ[i,j,k] = 1
                            cnt = occ_valid[i,j,k]
                            ptr_id = occ_index[i,j,k,:cnt]
                            interested_pcd.extend(ptr_id)
                            interested_idx_list.append([i,j,k])
        interested_pcd = np.array(interested_pcd)
        interested_idx_list = np.array(interested_idx_list)
        interested_pcd = interested_pcd[~mask[interested_pcd]]

    else:
        # find around interested_idx_list
        interested_occ = np.zeros_like(occ_valid)
        interested_pcd = []
        interested_idx_list_origin = interested_idx_list
        interested_idx_list = []
        tmp_querry_idx_set = set() # avoid to querry every voxel more than one time
        for idx in range(len(interested_idx_list_origin)):
            i_mean, j_mean, k_mean = interested_idx_list_origin[idx]
            for i in range(i_mean - 1,i_mean + 1):
                    for j in range(j_mean - 1, j_mean + 1):
                        for k in range(k_mean - 1, k_mean + 1):
                            if (i,j,k)  in tmp_querry_idx_set:
                                continue
                            else:
                                tmp_querry_idx_set.add((i,j,k))
                                # 2 types of occ is selected into occ_mask: 1.边缘：自身是1;上下左右存在0；2：边缘：自身是0上下左右存在1
                                flag1 = (occ_valid[i, j, k] > 0) and (occ_mask[i, j, k] == 0) and (
                                        (occ_mask[i + 1, j, k] > 0) or (occ_mask[i, j + 1, k] > 0) or (
                                        occ_mask[i, j, k + 1] > 0) or occ_mask[i - 1, j, k] > 0 or occ_mask[
                                            i, j - 1, k] > 0 or (occ_mask[i, j, k - 1] > 0))
                                flag2 = (occ_valid[i, j, k] > 0) and (occ_mask[i, j, k] == 1) and (
                                        (occ_mask[i + 1, j, k] == 0) or (occ_mask[i, j + 1, k] == 0) or (
                                        occ_mask[i, j, k + 1] == 0) or occ_mask[i - 1, j, k] == 0 or occ_mask[
                                            i, j - 1, k] == 0 or (occ_mask[i, j, k - 1] == 0))
                                if flag1 or flag2:  # all point is interesting
                                    interested_occ[i, j, k] = 1
                                    cnt = occ_mask[i, j, k]
                                    ptr_id = occ_index[i, j, k, :cnt]
                                    interested_pcd.extend(ptr_id)
                                    interested_idx_list.append([i, j, k])
        interested_pcd = np.array(interested_pcd)
        interested_idx_list = np.array(interested_idx_list)
        interested_pcd = interested_pcd[~mask[interested_pcd]]

    return interested_pcd, interested_idx_list, interested_occ





if __name__ == '__main__':
    #filter_neural_pcd(mask, pnt_neural_attr)
    pcd_t = o3d.t.io.read_point_cloud('/home/yuze/Documents/project/PC-NeRF/checkpoints/col_nerfsynth/lego_1023/edit/origin_lego_penerfneuralpoint.ply')
    msk = np.load('/home/yuze/Downloads/a.npy')
    pnt_neural_attr = read_neural_point('ripnerf',pcd_t)
    mask,mask_grown,num_of_decline, mask_decline = filter_decline_pcd(msk,pnt_neural_attr)

    pcd = pcd_t.to_legacy()
    color = np.zeros_like(pcd.colors).astype(np.uint8)
    # color [:,0] = 255
    # color[msk[:,0]] = [255,0,0]
    color[mask] = [0,255,0]
    color[mask_decline] = [255,0,0]
    pcd.colors = o3d.utility.Vector3dVector(color)

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Visualizer", 800, 800)
    vis.add_geometry(pcd)
    while True:
        vis.poll_events()
        vis.update_renderer()