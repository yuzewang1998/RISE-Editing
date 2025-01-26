import numpy as np

def build_occ(ptr,voxel_size = np.array([0.016,0.016,0.016]),max_ptr_in_a_occ = 32):
    '''
    ptr: np.array [n,3]
    '''
    bbox = cauc_bounding_box(ptr)
    occ_shift =bbox[:,0]-1e-6
    shifted_ptr = ptr-occ_shift#[n,3]
    shifted_ptr_occ_coord = np.floor(shifted_ptr/voxel_size).astype(int) #[n,3] int
    occ_shape = np.ceil((bbox[:,1]-bbox[:,0])/voxel_size).astype(int)
    occ_index = np.zeros([occ_shape[0],occ_shape[1],occ_shape[2],max_ptr_in_a_occ],dtype=np.int32)#e.g[60,50,40,32]
    occ_mask = np.zeros(occ_shape,dtype=np.int32)
    for i in range(len(ptr)):
        coord_x,coord_y,coord_z = shifted_ptr_occ_coord[i]#[3,]
        occ_index[coord_x,coord_y,coord_z,occ_mask[coord_x,coord_y,coord_z]] = i
        occ_mask[coord_x,coord_y,coord_z] += 1
    return occ_index,occ_mask

def cauc_bounding_box(ptr):
    '''
    Caucaulate bounding box of a point cloud
    ptr:n*3
    return : [3,2]
    '''
    lower = np.min(ptr,axis=0)
    upper = np.max(ptr, axis=0)
    bbox = np.concatenate([lower[...,None],upper[...,None]],axis=-1)
    return bbox

if __name__=="__main__":
    ptr = np.random.rand(1000,3)
    build_occ(ptr)