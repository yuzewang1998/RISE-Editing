# RISE-Editing: Rotation-invariant Neural Point Fields with Interactive Segmentation for Fine-grained and Efficient Editing
RISE-Editing uses neural 3D point clouds, with associated neural features, to model and editing radiance field. Specifically, we first propose a novel rotation-invariant neural point field representation to learn local contents with the Cartesian coordinates, enhancing the rendering quality after fine-grained editing of the reconstructed scenes. Secondly, we present a plug-and-play multi-view ensemble learning strategy to lift the 2D inconsistent zero-shot segmentation results to 3D neural point field without post retraining. Users can efficiently achieve the interactive implicit field segment-then-editing with this strategy. Third, we propose an efficient and fine-grained neural scene shape editing framework. With simple click-based prompts, users can segment the implicit point field they want to edit, and generate novel implicit field with variopus shape editing functions in real-time, such as part transformation, duplication, scaling up or down, and cross-scene compositing.
![image](https://github.com/yuzewang1998/RISE-EDITING/blob/master/assets/teaser.jpg)
## Overal Instruction
1. Please first install the libraries as below and download/prepare the datasets as instructed.
2. Point Initialization: Download pre-trained MVSNet as below and train the feature extraction from scratch or directly download the pre-trained models. (Obtain 'MVSNet' and 'init' folder in checkpoints folder)
3. Per-scene Optimization: Optimize from scratch as instructed.
4. Select neural point with SAM (prompt from multi-view images) or MeshLab (prompt from 3D point cloud)
5. Editing neural point ( local transformation, duplication, deletion, scaling up or down, and cross-scene compositing)

## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 22.04)
* Python 3.6+
* PyTorch 1.7 or higher
* CUDA 10.2 or higher 

### Install
Install the dependent libraries as follows:

* Install the dependent python libraries:
```
pip install torch==1.8.1+cu102 h5py
pip install imageio scikit-image
```

* Install pycuda (crucial) following:
https://documen.tician.de/pycuda/install.html 
* Install torch_scatter following:
https://github.com/rusty1s/pytorch_scatter
* (Optional) If you want to edit radiance field via 3D neural points scaffold with MeshLab, just skip this step. If you want to edit via prompting 2D images with SAM, install [SAM](https://github.com/facebookresearch/segment-anything) and put it at ```./segment-anything``` folder and download checkpoints at ```./segment-anything/checkpoints```( Or you can customize the SAM path in ```./Editor/SAMER/app.py```)

We develope our code with pytorch1.8.1, pycuda2021.1, and torch_scatter 2.0.8.

## Data Preparation

The layout should looks like this
```
rise-editing
├── data_src
│   ├── dtu
    │   │   │──Cameras
    │   │   │──Depths
    │   │   │──Depths_raw
    │   │   │──Rectified
    ├── nerf
    │   │   │──nerf_synthetic
    │   │   │──nerf_synthetic_colmap
    ├── TanksAndTemple
    ├── scannet
    │   │   │──scans 
    |   │   │   │──scene0101_04
    |   │   │   │──scene0241_01
```
Or you can download using the official links as follows:


### NeRF Synthetic dataset
Download NeRF Synthetic dataset with COLMAP from [here](https://drive.google.com/drive/folders/1sonP_5JLNE5c5ejHpqke_7M84Zd4O7eH) under ``data_src/nerf/``
### Scannet dataset
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/. The detailed steps including:
* Go to http://www.scan-net.org and fill & sent the request form.
* You will get a email that has command instruction and a download-scannet.py file, this file is for python 2, you can use our download-scannet.py in the ``data'' directory for python 3.
* clone the official repo:
    ```
    git clone https://github.com/ScanNet/ScanNet.git
    ```
* Download specific scenes, such as:
  ```
   python data/download-scannet.py -o ../data_src/scannet/ id scene0113_01
  ```
* Process the sens files:
  ```
    python ScanNet/SensReader/python/reader.py --filename data_src/nrData/scannet/scans/scene01113_01/scene01113_01.sens  --output_path data_src/nrData/scannet/scans/scene01113_01/exported/ --export_depth_images --export_color_images --export_poses --export_intrinsics
  ```

## Point Initialization / Generalization:
### &nbsp; Download pre-trained MVSNet checkpoints:
We trained [MVSNet](https://github.com/xy-guo/MVSNet_pytorch) on DTU. You can Download ''MVSNet'' directory from 
[google drive](https://drive.google.com/drive/folders/1dVAWn5j3e8JmHJSjr4csPP_jUsOZVhnL?usp=drive_link)
and place them under ```./checkpoints/```

### &nbsp;  Train 2D feature extraction and point representation
#####  &nbsp; Directly use our trained checkpoints files:
Download ```init``` directory from 
[google drive](https://drive.google.com/drive/folders/1Y_AxuUgkAFmnskb25u8AOWgeQYp55xjh?usp=drive_link).
and place them under ```./checkpoints/```

##### &nbsp; Or train from scratch:
Train for point features of 63 channels (as in paper) 
```
bash dev_scripts/ete/dtu_dgt_d012_img0123_conf_color_dir_agg2.sh
```
Train for point features of 32 channels (better for per-scene optimization)
```
bash dev_scripts/ete/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20.sh
```
After the training, you should pick a checkpoint and rename it to best checkpoint, e.g.:
```
cp checkpoints/dtu_dgt_d012_img0123_conf_color_dir_agg2/250000_net_ray_marching.pth  checkpoints/dtu_dgt_d012_img0123_conf_color_dir_agg2/best_net_ray_marching.pth

cp checkpoints/dtu_dgt_d012_img0123_conf_color_dir_agg2/250000_net_mvs.pth  checkpoints/dtu_dgt_d012_img0123_conf_color_dir_agg2/best_net_mvs.pth
```

### Download per-scene optimized 
We upload various editing case (some has been shown in the paper, some not). You can skip training and download the folders here [google drive](https://drive.google.com/drive/folders/12zJbFvV80YVm63IPFzqnxKXnbkgKLeWf?usp=sharing), and place them in ```checkpoints/```.

```
rise-editing
├── checkpoints
│   ├── init
    ├── MVSNet
    ├── col_nerfsynth
    ├── scannet
    ...
```
### Train the per-scene RISE-Editing
Make sure the ''checkpoints'' folder has ''init'' and ''MVSNet''.
The training scripts will start to do initialization if there is no ''.pth'' files in a scene folder. It will start from the last ''.pth'' files until reach the iteration of ''maximum_step''.
 For example:
```
    bash dev_scripts/w_colmap_n360/chair.sh
    bash dev_scripts/w_colmap_n360/drums.sh
    bash dev_scripts/w_scannet_etf/scene113.sh
    bash dev_scripts/w_scannet_etf/scene33.sh
```
### Train across scene for compositing
We provide an example for simitaneously training multiple scenes for compositing:
```aiignore
./run/train_ft_ms.py
```
### Test the per-scene optimized RISE-Editing

  For example:
```
    bash dev_scripts/w_colmap_n360/chair_test.sh
    bash dev_scripts/w_colmap_n360/drums_test.sh
    bash dev_scripts/w_scannet_etf/scene113_test.sh
    bash dev_scripts/w_scannet_etf/scene33_test.sh
```
### Prompting and editing the neural points you want to edit.
#### Option 1: Prompting the neural points with MeshLab
We provide several editing example scripts in ``./Editor/editor_run`` folder.
It mainly contains four steps:
* extract the neural points explicitly (.ply) from the checkpoints (xxx_net_ray_marching.pth). A ``*_neuralpoint.ply`` will generated.
* Open the ``*_neural_point.ply`` with Meshlab, and select the neural points you want to edit, export it as '*_meshlabpoint.ply'
* convert the meshlab type point to neural type point.
* Manipulate the corresponding neural point (transformation, scaling up/down, deletion, composition across scenes...)
* replace the generate ``*_net_ray_marching.pth`` to the checkpoints before editing (or just rename it as +1 iteration) 
* Render the edited scene with the test script, such as ``./dev_scripts/w_colmap_n360/ship_test.sh``

We provide an example of ship editing: select the ship at the top of the platte, duplicate it twice, and position the duplicates to the left and right of the original ship.
The following image illustrates the selected body of the ship:
![image](https://github.com/yuzewang1998/RISE-EDITING/blob/master/assets/edit_meshlab3.png)
The following image illustrates the duplicated ship and the original ship:
![image](https://github.com/yuzewang1998/RISE-EDITING/blob/master/assets/edit_meshlab2.png)
The following image illstrates the edited scene:
![image](https://github.com/yuzewang1998/RISE-EDITING/blob/master/assets/edit_meshlab1.png)
The following image illustrates the rendering results:
![image](https://github.com/yuzewang1998/RISE-EDITING/blob/master/assets/render.png)
The following image illustrates the workspace file of the directory:
![image](https://github.com/yuzewang1998/RISE-EDITING/blob/master/assets/file_directory.png)
#### Option 2: Prompting the neural points with SAM
The mainly difference is the select neural points step. We use a GUI to interactive segment the part we want to edit with SAM:
```python ./Editor/SAMER/app.py```
Then you will get the neural point you want to edit. And the following steps are same as the MeshLab one.
## Acknowledgement
Our repo is developed based on [Point-NeRF](https://github.com/Xharlie/pointnerf), [SAM](https://github.com/facebookresearch/segment-anything) and [SPIDR](https://github.com/nexuslrf/SPIDR)

Please also consider citing the corresponding papers. 

## LICENSE
The repo is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 2.0, and is restricted to academic use only.
See [LICENSE](https://github.com/Xharlie/pointnerf/blob/master/LICENSE.md).
