import importlib
from Editor.points.base_point import BasePoint
import numpy as np
#对model的多继承,选择合适的model进行import
def find_neuralpoint_class_by_name(neuralpoint_name):
    # PointNerf
    # the file "models/modelname_model.py"
    # will be imported.
    neural_point_file_name = "Editor.points." + neuralpoint_name+"_neural_point" #In general: mvs_points_volumetric_model
    modellib = importlib.import_module(neural_point_file_name)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = (neuralpoint_name+"_neural_point" ).replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BasePoint):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseNeuralPoint with class name that matches %s in lowercase."
            % (neural_point_file_name, target_model_name))
        exit(0)
    return model

def create_neural_point(opt,classname):
    neural_point = find_neuralpoint_class_by_name(classname)
    instance = neural_point(opt)
    return instance

def find_neural_point_id(meshlab_ptr,neural_ptr,voxel_size = np.array([0.008,0.008,0.008]),max_ptr_in_a_occ = 64):
    '''
    ptr: np.array [n,3]
    '''
    bbox = cauc_bounding_box(neural_ptr)
    occ_shift =bbox[:,0]-1e-9
    shifted_neural_ptr = neural_ptr-occ_shift#[n,3]
    shifted_neural_ptr_occ_coord = np.floor(shifted_neural_ptr/voxel_size).astype(int) #[n,3] int
    shifted_meshlab_ptr_occ_coord = np.floor((meshlab_ptr-occ_shift)/voxel_size).astype(int)
    occ_shape = np.ceil((bbox[:,1]-bbox[:,0])/voxel_size).astype(int)
    occ_index = np.zeros([occ_shape[0],occ_shape[1],occ_shape[2],max_ptr_in_a_occ],dtype=np.int32)#e.g[60,50,40,32]
    occ_mask = np.zeros(occ_shape,dtype=np.int32)
    # build occ
    for i in range(len(neural_ptr)):
        coord_x,coord_y,coord_z = shifted_neural_ptr_occ_coord[i]#[3,]
        occ_index[coord_x,coord_y,coord_z,occ_mask[coord_x,coord_y,coord_z]] = i
        occ_mask[coord_x,coord_y,coord_z] += 1
    # quick search
    selected_id_list = []
    for i in range(len(shifted_meshlab_ptr_occ_coord)):
        coord_x,coord_y,coord_z = shifted_meshlab_ptr_occ_coord[i]
        nums_of_neural_ptr = occ_mask[coord_x,coord_y,coord_z]
        selected_neural_ptr_id = occ_index[coord_x,coord_y,coord_z,:nums_of_neural_ptr]
        selected_neural_ptr_xyz = neural_ptr[selected_neural_ptr_id]
        dist_msk =np.sqrt(np.sum(np.square(selected_neural_ptr_xyz - meshlab_ptr[i]), axis=-1))<1e-9
        selected_id_list.extend(selected_neural_ptr_id[dist_msk])
    return selected_id_list

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

if __name__ == "__main__":
    pass