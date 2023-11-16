import cv2
import imageio
import numpy as np
import argparse
import os
import open3d as o3d
import json
from point_utils import project_pcd, get_depth_map
from projection import find_project_function
from fusion import find_fusion_function
from fusion.neural_point_utils import read_neural_point, filter_neural_pcd

'''
interactive mode : 在'e'时候做区域增长
debug mode :mask循环完了做区域增长
'''

parser = argparse.ArgumentParser()
# True False
# The most important is Model Ensemble
parser.add_argument("--debug_mode",type=bool,default=False,help='If switch on debug mode, the prompt will read from txt rather than user interactive input')
parser.add_argument("--save_akp",type=bool,default=True,help='save annotated keypoints if on')
parser.add_argument("--akp_json_path",type=str,default="/home/yuze/Documents/project/PC-NeRF/checkpoints/col_nerfsynth/lego/edit/anotated_keypoints.json",help='If save_akp is on, this is the save akp path.If debug mode is on, read the json file for annotated keypoints for each frame')
parser.add_argument("--grow_region",type=bool,default=True)
parser.add_argument("--decline_region",type=bool,default=False)
parser.add_argument("--image", type=str, default="demo.png")
parser.add_argument("--wo_sam", action="store_true")
parser.add_argument("--save_path", type=str, default="/home/yuze/Documents/project/PC-NeRF/checkpoints/col_nerfsynth/lego/edit/")

parser.add_argument("--dataset_path", type=str, default=r"/home/yuze/Documents/project/PC-NeRF/data_src/nerf/nerf_synthetic/lego")
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument("--dataset_skip", type=int, default=10)

parser.add_argument("--pcd_path", type=str, default=r"/home/yuze/Documents/project/PC-NeRF/checkpoints/col_nerfsynth/lego/edit/origin_lego_penerfneuralpoint.ply")


parser.add_argument('-which_projection_function',type=str, default = 'mask_projection', help='mask_projection |')
parser.add_argument('-which_fusion_function',type=str, default = 'depth_logits_distanceweightedaverage_fusion', help='simple_mask_logical_fusion | depth_mask_logical_fusion |simple_logits_xxx_fusion|simple_logits_logitsaverage_fusion|depth_logits_logitsaverage_fusion|simple_mask_voting_fusion|depth_mask_voting_fusion|depth_logits_distanceweightedaverage_fusion|udepth_logits_distanceweightedaverage_fusion')
parser.add_argument('-which_neuralpoint',type=str,default='ripnerf',help='ripnerf|standard')
# TODO: Neural Point color is so ugly...
# initialize the settings
args = parser.parse_args()
args.use_sam = not args.wo_sam


"""
FUSION_MODE: core of the algorithm
simple_mask_logical_fusion:  project every point; user operates 'AND' + 'OR'
depth_mask_logical_fusion:  project with depth( GT)?
"""
fusion_func = find_fusion_function(args.which_fusion_function)
if args.which_fusion_function.split('_')[1] == 'mask':
    args.which_projection_function = 'mask_projection'
elif args.which_fusion_function.split('_')[0] == 'udepth' and args.which_fusion_function.split('_')[1].startswith('logits'):
    args.which_projection_function = 'logits_uncertain_projection'
elif args.which_fusion_function.split('_')[1].startswith('logits'):
    args.which_projection_function = 'logits_projection'

# initialize the renderer
project_func = find_project_function(args.which_projection_function)
cv2.namedWindow("2D Annotator")

# initialize some varibles
vis = None
pnt_w = None
pnt_frame_buffer = []
# pnt_frame_mask = None
pnt_mask_curr = None

# initialize the SAM
if args.use_sam:
    import sys
    sys.path.append("/home/yuze/anaconda3/envs/pointnerf/lib/python3.8/site-packages/segment_anything")
    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = "/home/yuze/anaconda3/envs/pointnerf/lib/python3.8/site-packages/segment_anything/checkpoints/sam_vit_h_4b8939.pth"
    # sam_checkpoint = "segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    sam.to("cuda")

    predictor = SamPredictor(sam)
else:
    predictor = None

obj_size = 4
obj_mode = 'pcd'
# Set 3D point cloud
if args.pcd_path != "":
    pcd_t = o3d.t.io.read_point_cloud(args.pcd_path)
    pcd = pcd_t.to_legacy()
    pnt_w = np.asarray(pcd.points)
    # set a pcd attribute list
    pnt_neural_attr = read_neural_point(args.which_neuralpoint,pcd_t)
    # read rip-nerf neuralpoint embedding at the moment
    color_ori = np.clip((np.asarray(pcd.colors).copy()*256-128),0,255).astype(np.uint8)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
    obj_size = np.max(bbox_size)
    pcd.colors = o3d.utility.Vector3dVector(color_ori)
    obj = pcd
else:
    exit(1)

vis = o3d.visualization.Visualizer()
vis.create_window("3D Visualizer", 800, 800)

vis.add_geometry(obj)




# Initialize the list of keypoints
keypoints = []
akp_dict_data = []
# Colors for different modes
colors = [(0, 0, 255), (0, 255, 0)]
pnt_sel_color = np.array([255, 0, 0])
pnt_sel_color_global = np.array([255, 0, 0])
pnt_mask_idx = -1
# Initialize the mode
mode = 1
sel_mode = 1 # 0: single frame, 1: multi frame

sam_mode = 0
"""
SAM_MODE
0: SINGLE OUTPUT WITH BITMAP [DEFAULT]
1. SINGLE OUTPUT WITH LOGITS
e.g. Multi mask output or automatic mask output don't work well.
"""

or_and_mode = 0
"""
Only used in FUSION_MODE:logical_fusion
OR:0; AND:1
"""
# Initialize the mask transparency
depth_ratio = 10


depth_frames = []
image_idx = 0
if args.dataset_path != "":
    from nerf_synthetic import NeRFSynthetic
    data = NeRFSynthetic(args.dataset_path, split=args.dataset_split, testskip=args.dataset_skip)
    n_images = len(data)
    original_image_rgb, c2w, image_path = data[image_idx]
    if pnt_w is not None:
        uv_cam, pnt_cam, depth_pnt = project_pcd(pnt_w, data.K, c2w)
        depth_map, index = get_depth_map(uv_cam, depth_pnt, *original_image_rgb.shape[:2], scale=3)

else:
    # Load the image
    image_path = args.image
    # original_image = cv2.imread(image_path)
    # original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_rgb = imageio.imread(image_path)
    if original_image_rgb.shape[-1] == 4:
        original_image_rgb = original_image_rgb / 255.
        original_image_rgb = original_image_rgb[:,:,:3] * original_image_rgb[:,:,3:4] + (1 - original_image_rgb[:,:,3:4])
        original_image_rgb = (original_image_rgb.clip(0, 1) * 255).astype(np.uint8)

original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
if predictor is not None:
    predictor.set_image(original_image_rgb)
image = original_image.copy()
logits = None
mask = None
print("Image loaded")

if not args.debug_mode:
    # Mouse callback function
    def annotate_keypoints_singleview(event, x, y, flags, param):
        global keypoints, mode, image, logits, mask,sam_mode

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the keypoint and mode to the list
            keypoints.append((x, y, mode))
            # print("Keypoint added:", (x, y, mode))
            if predictor is not None:
                    # Run SAM

                    input_point = np.array([pts[:2] for pts in keypoints])
                    input_label = np.array([pts[2] for pts in keypoints])
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        mask_input=logits,
                        multimask_output=False,
                    )
                    mask = masks[0]
                    # render~
                    color_mask = (np.random.random(3) * 255).astype(np.uint8)
                    colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color_mask
                    image = cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)

            else:
                image = original_image.copy()

            # Draw a circle at the keypoint position with the corresponding color
            for x, y, m in keypoints:
                cv2.circle(image, (x, y), 3, colors[m], -1)
                cv2.imshow("2D Annotator", image)
    # Mouse callback function

    # Trackbar callback function
    def on_trackbar(val):
        global depth_ratio
        depth_ratio = val
        # update_image()
        cv2.imshow("2D Annotator", image)
    # Create a window and set the mouse callback function
    cv2.setMouseCallback("2D Annotator", annotate_keypoints_singleview)
    # Create a trackbar (slider) to control the depth_ratio of the mask
    cv2.createTrackbar("Depth Percentage", "2D Annotator", depth_ratio, 100, on_trackbar)

    print("Start annotating keypoints")
    while True:
        cv2.imshow("2D Annotator", image)
        key = cv2.waitKey(1) & 0xFF
        # Implement myself
        if key == ord("o"):
            or_and_mode = (or_and_mode + 1) % 2
        # Press 'm' to toggle between modes
        if key == ord("m"):
            mode = (mode + 1) % 2
        # Press 'z' to undo the last keypoint
        if key == ord("z"):
            if keypoints:
                # Remove the last keypoint
                keypoints.pop()

                # Redraw the keypoints
                image = original_image.copy()
                for x, y, m in keypoints:
                    cv2.circle(image, (x, y), 3, colors[m], -1)
                cv2.imshow("2D Annotator", image)
        # Press 's' to save the mask and keypoints
        if key == ord("s"):
            image_name = os.path.basename(image_path)
            if mask is not None:
                os.makedirs(args.save_path, exist_ok=True)
                mask_path = os.path.join(args.save_path, image_name)
                imageio.imwrite(mask_path + '.png', (mask[..., None]*255).astype(np.uint8))
                np.savetxt(mask_path + ".txt", keypoints, fmt="%d")
        # Press 'n' to go to the next image
        if key == ord("n"):
            image_idx = (image_idx + 1) % n_images
            if args.dataset_path != "":
                original_image_rgb, c2w, image_path = data[image_idx]
                original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
                if pnt_w is not None:
                    uv_cam, pnt_cam, depth_pnt = project_pcd(pnt_w, data.K, c2w)
                    depth_map, index = get_depth_map(uv_cam, depth_pnt, *original_image_rgb.shape[:2], scale=3)

                if predictor is not None:
                    predictor.set_image(original_image_rgb)
                image = original_image.copy()
                keypoints = []
                logits = None
                mask = None
        # Press 'p' to go to the previous image
        if key == ord("p"):
            image_idx = (image_idx - 1) % n_images
            if args.dataset_path != "":
                original_image_rgb, c2w, image_path = data[image_idx]
                original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
                if pnt_w is not None:
                    uv_cam, pnt_cam, depth_pnt = project_pcd(pnt_w, data.K, c2w)
                    depth_map, index = get_depth_map(uv_cam, depth_pnt, *original_image_rgb.shape[:2], scale=3)

                if predictor is not None:
                    predictor.set_image(original_image_rgb)
                image = original_image.copy()
                keypoints = []
                logits = None
                mask = None
        # Press 'r' to reset the image
        if key == ord("r"):
            image = original_image.copy()
            keypoints = []
            logits = None
            mask = None
            cv2.imshow("2D Annotator", image)
        # Press 'c' to crop the point cloud
        if key == ord("c") and pnt_w is not None and mask is not None:
            # project and fusion
            fusion_func_kwargs = {}
            if args.which_fusion_function.split('_')[2] == 'logical':
                fusion_func_kwargs['or_and_mode'] = or_and_mode
            if args.which_fusion_function.split('_')[0].endswith('depth'):
                fusion_func_kwargs['depth_pnt'] = depth_pnt
                depth_thresh = obj_size * depth_ratio / 100
                fusion_func_kwargs['depth_thresh'] = depth_thresh
            if args.which_fusion_function.split('_')[1].startswith('logits'):
                fusion_func_kwargs['logits'] = logits
            if args.which_fusion_function.split('_')[1] == 'logitsAndEmbedding':
                fusion_func_kwargs['pnt_neural_attr'] = pnt_neural_attr
            # fusion the masked pcd
            pnt_frame_buffer, pnt_mask_curr = fusion_func(pnt_mask_curr,uv_cam,mask,project_func,pnt_frame_buffer,image_idx,c2w,depth_map,**fusion_func_kwargs)
            # 2D renderer
            color = (~pnt_mask_curr) * color_ori + pnt_mask_curr * pnt_sel_color / 225
            if args.save_akp:
                akp_itm = {
                    "image_idx" : image_idx,
                    "keypoints": keypoints,
                }
                akp_dict_data.append(akp_itm)
                print('Append image_idx .{} akp'.format(image_idx))
            obj.colors = o3d.utility.Vector3dVector(color)
            # 3D renderer
            vis.update_geometry(obj)


        # Press 'e' to export the masked point cloud
        if key == ord("e") and pnt_w is not None and pnt_mask_curr is not None:
                # Accomplish the fusion, do region growing with embeddings
                if args.grow_region:
                    number_of_grown = 999999
                    grow_iter = 0
                    interested_idx_list = []
                    while number_of_grown > 50 and grow_iter < 5:
                        print('growing region....,iter:{}'.format(grow_iter))
                        pnt_mask_curr, number_of_grown, interested_idx_list, _ = filter_neural_pcd(pnt_mask_curr,
                                                                                                   pnt_neural_attr,
                                                                                                   interested_idx_list)

                        print('iter:{} has been growned, {} points has been grown'.format(grow_iter, number_of_grown))
                        grow_iter += 1
                # Not every scene work well
                if args.decline_region:
                    number_of_decline = 99999
                    decline_iter = 0
                    while number_of_decline > 5 and decline_iter < 5:
                        print('growing region....,iter:{}'.format(grow_iter))

                color = (~pnt_mask_curr) * color_ori + pnt_mask_curr * pnt_sel_color / 225

                obj.colors = o3d.utility.Vector3dVector(color)
                # 3D renderer
                vis.update_geometry(obj)

                pcd_t.point.flags = (pnt_mask_curr * 32).astype(np.int32) # this is to nb
                mask_pcd_name = os.path.basename(args.pcd_path)[:-4] + '_mask.ply'
                sel_pcd_name = os.path.basename(args.pcd_path)[:-4] + '_sel.ply'
                o3d.t.io.write_point_cloud(os.path.join(args.save_path, mask_pcd_name), pcd_t)

                pnt_mask_idx = np.where(pnt_mask_curr)[0].astype(np.int64)
                selected_pcd = pcd_t.select_by_index(pnt_mask_idx)
                os.makedirs(args.save_path, exist_ok=True)
                o3d.t.io.write_point_cloud(os.path.join(args.save_path, sel_pcd_name), selected_pcd)

                print('Export masked point cloud to', os.path.join(args.save_path, sel_pcd_name))
                if args.save_akp:
                    # akp_dict_data = json.dumps(akp_dict_data)
                    with open(args.akp_json_path, "w") as json_file:
                        json.dump(akp_dict_data, json_file)
        # Press 'q' to exit
        if key == ord("q"):
            break

        if vis is not None:
            vis.poll_events()
            vis.update_renderer()

    # Close all windows
    cv2.destroyAllWindows()
    if vis is not None:
        vis.destroy_window()
else:
    # debug_akp_json_path
    # load annotated kp data (multiple frame)

    with open(args.akp_json_path, "r") as json_file:
        akp_dict_data = json.load(json_file)
        print(akp_dict_data)
        # loop for each frame, each kp
        for akp_data in akp_dict_data:
            keypoints = akp_data['keypoints']
            image_idx = akp_data['image_idx']
            # init img and SAM_model
            if args.dataset_path != "":
                original_image_rgb, c2w, image_path = data[image_idx]
                original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
                if pnt_w is not None:
                    uv_cam, pnt_cam, depth_pnt = project_pcd(pnt_w, data.K, c2w)
                    depth_map, index = get_depth_map(uv_cam, depth_pnt, *original_image_rgb.shape[:2], scale=3)

                if predictor is not None:
                    predictor.set_image(original_image_rgb)
                image = original_image.copy()
            # predict msk
            if predictor is not None:
                # Run SAM
                input_point = np.array([pts[:2] for pts in keypoints])
                input_label = np.array([pts[2] for pts in keypoints])
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=logits,
                    multimask_output=False,
                )
                mask = masks[0]
            # project and fusion
            fusion_func_kwargs = {}
            if args.which_fusion_function.split('_')[2] == 'logical':
                fusion_func_kwargs['or_and_mode'] = or_and_mode
            if args.which_fusion_function.split('_')[0].endswith('depth'):

                fusion_func_kwargs['depth_pnt'] = depth_pnt
                depth_thresh = obj_size * depth_ratio / 100
                fusion_func_kwargs['depth_thresh'] = depth_thresh
            if args.which_fusion_function.split('_')[1].startswith('logits'):
                fusion_func_kwargs['logits'] = logits
            if args.which_fusion_function.split('_')[1] == 'logitsAndEmbedding':
                fusion_func_kwargs['pnt_neural_attr'] = pnt_neural_attr
            # fusion the masked pcd
            pnt_frame_buffer, pnt_mask_curr = fusion_func(pnt_mask_curr,uv_cam,mask,project_func,pnt_frame_buffer,image_idx,c2w,depth_map,**fusion_func_kwargs)
        # Accomplish the fusion, do region growing with embeddings
        if args.grow_region:
            number_of_grown = 999999
            grow_iter = 0
            interested_idx_list = []
            while number_of_grown > 50 and grow_iter<5:
                print('growing region....,iter:{}'.format(grow_iter))
                pnt_mask_curr, number_of_grown, interested_idx_list,_ = filter_neural_pcd(pnt_mask_curr, pnt_neural_attr, interested_idx_list)

                print('iter:{} has been growned, {} points has been grown'.format(grow_iter,number_of_grown))
                grow_iter+=1
        # Not every scene work well
        if args.decline_region:
            number_of_decline = 99999
            decline_iter = 0
            while number_of_decline > 5 and decline_iter < 5:
                print('growing region....,iter:{}'.format(grow_iter))

        pcd_t.point.flags = (pnt_mask_curr * 32).astype(np.int32)
        mask_pcd_name = os.path.basename(args.pcd_path)[:-4] + '_mask.ply'
        sel_pcd_name = os.path.basename(args.pcd_path)[:-4] + '_sel.ply'
        o3d.t.io.write_point_cloud(os.path.join(args.save_path, mask_pcd_name), pcd_t)

        pnt_mask_idx = np.where(pnt_mask_curr)[0].astype(np.int64)
        selected_pcd = pcd_t.select_by_index(pnt_mask_idx)
        os.makedirs(args.save_path, exist_ok=True)
        o3d.t.io.write_point_cloud(os.path.join(args.save_path, sel_pcd_name), selected_pcd)
        print('Export masked point cloud to', os.path.join(args.save_path, sel_pcd_name))

# Print the annotated keypoints
print("Annotated keypoints:", keypoints)
