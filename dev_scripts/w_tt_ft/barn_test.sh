#!/bin/bash
nrCheckpoint="../checkpoints"
nrDataRoot="../data_src"
name='barn'

resume_iter=latest # 20000 #latest
data_root="${nrDataRoot}/TanksAndTemple/"
scan="Barn"

normview=0
point_conf_mode="1" # 0 for only at features, 1 for multi at weight
point_dir_mode="1" # 0 for only at features, 1 for color branch
point_color_mode="1" # 0 for only at features, 1 for color branch
mvs_img_wh=" 1088 640 "
img_wh=" 1088 640 "

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=15
radius_limit_scale=4
depth_limit_scale=0
alpha_range=1

vscale=" 3 3 3 "
kernel_size=" 3 3 3 "
query_size=" 3 3 3 "
vsize=" 0.003 0.003 0.003 " #" 0.005 0.005 0.005 "
wcoord_query=1
z_depth_dim=400
max_o=1500000 #2000000
ranges=" -2.05965 -0.48064 -2.23660 1.78036 0.6094 1.28341 "
SR=40
K=8
P=11
NN=2


act_type="LeakyReLU"
agg_intrp_order=2
agg_distance_kernel="linear_immediately" #"avg" #"feat_intrp"
weight_xyz_freq=2
weight_feat_dim=8
point_features_dim=32
shpnt_jitter="uniform" #"uniform" # uniform gaussian

which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer0=0
shading_feature_mlp_layer1=2
shading_feature_mlp_layer2=0
shading_feature_mlp_linear=2
shading_feature_mlp_layer3=0 #0
shading_feature_mlp_layer4=2 #1
shading_feature_mlp_layer0_rotation_invariance_feature_extraction_module=0
shading_feature_mlp_layer0_rotation_invariance_feature_extraction_dim=999
shading_alpha_mlp_layer=1
shading_color_mlp_layer=2
shading_feature_num=256
dist_xyz_freq=5
num_feat_freqs=3
dist_xyz_deno=0

raydist_mode_unit=1
dataset_name='tt_ft'
pin_data_in_memory=0
model='mvs_points_volumetric'
near_plane=0.0
far_plane=4.5
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #
domain_size='1'
dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4
num_pos_freqs=10
num_viewdir_freqs=4 #6
random_sample='random'
random_sample_size=48 #48 # 32 * 32 = 1024

batch_size=1
gpu_ids='0'

checkpoints_dir="${nrCheckpoint}/tanksntemples/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20"
test_num_step=1
visual_items=' coarse_raycolor gt_image '

color_loss_weights=" 1.0 0.0 0.0 "
color_loss_items='ray_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor'
test_color_loss_items='coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor'

bg_color="white" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"

cd run

python3 test_ft.py \
        --experiment $name \
        --scan $scan \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --gpu_ids $gpu_ids \
        --checkpoints_dir $checkpoints_dir \
        --pin_data_in_memory $pin_data_in_memory \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --bg_color $bg_color \
        --split $split \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer0 $shading_feature_mlp_layer0 \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_linear $shading_feature_mlp_linear \
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_mlp_layer4 $shading_feature_mlp_layer4 \
        --shading_feature_mlp_layer0_rotation_invariance_feature_extraction_module $shading_feature_mlp_layer0_rotation_invariance_feature_extraction_module \
        --shading_feature_mlp_layer0_rotation_invariance_feature_extraction_dim $shading_feature_mlp_layer0_rotation_invariance_feature_extraction_dim \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --color_loss_weights $color_loss_weights \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --visual_items $visual_items \
        --act_type $act_type \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --normview $normview \
        --alpha_range $alpha_range \
        --ranges $ranges \
        --mvs_img_wh $mvs_img_wh \
        --img_wh $img_wh \
        --vsize $vsize \
        --wcoord_query $wcoord_query \
        --max_o $max_o \
        --debug

