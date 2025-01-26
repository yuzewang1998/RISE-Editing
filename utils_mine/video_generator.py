import sys
import os
import pathlib
import argparse
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '../../NSEPN'))
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from utils.util import to8b
import imageio
'''
README:
1. 处理的是scannet这个数据集，文件组织遵循point-nerf；
'''
class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Demo of argparse")
        parser.add_argument('--data_root',type=str, default='/home/yuze/Documents/project/PC-NeRF/checkpoints/col_nerfsynth/lego/test_500001/images',help='root of rendering result(It is probably in checkpoints file)')
        # parser.add_argument('--unit',type=str, default='pose',choices=['iter','pose'],help='how to generate video,iter means show the fist pic every several iters,pose mean show the latest iter every camera poss')
        #parser.add_argument('--num_iter', type=int,default=10000,help='n iter you want to generator')
        parser.add_argument('--video_format', type=str, default='gif',choices=['mp4','gif','mov'],help='video format')
        parser.add_argument('--fps', type=int, default=20,help='frame per second of video')

        self.opt = parser.parse_args()

        # print(self.opt.dataset_dir)

class VideoGenerator:
    def __init__(self,opt):
        self.data_root = opt.data_root
        self.video_format = opt.video_format
        self.fps = opt.fps

        render_path_list = [i for i in os.listdir(self.data_root) if i.endswith("-coarse_raycolor.png")]
        render_path_list.sort()
        self.render_path_list = [os.path.join(self.data_root,img_path) for img_path in render_path_list]

        gt_path_list = [i for i in os.listdir(self.data_root) if i.endswith("-gt_image.png")]
        gt_path_list.sort()
        self.gt_path_list = [os.path.join(self.data_root,img_path) for img_path in gt_path_list]
    def gen_video(self):
        render_img_lst = []
        for img_path in tqdm(self.render_path_list):
            image = np.asarray(Image.open(img_path))
            render_img_lst.append(image)
        video_save_path = os.path.join(self.data_root,'video_render'+'.'+self.video_format)
        print("Generating render video...")
        imageio.mimwrite(video_save_path, render_img_lst, fps=self.fps)
        print("Generating render video done at ",video_save_path)

        gt_img_lst = []
        for img_path in tqdm(self.gt_path_list):
            image = np.asarray(Image.open(img_path))
            gt_img_lst.append(image)
        video_save_path = os.path.join(self.data_root,'video_gt'+'.'+self.video_format)
        print("Generating gt video...")
        imageio.mimwrite(video_save_path, gt_img_lst, fps=self.fps)



def main():
    sparse = Options()
    opt = sparse.opt
    print(opt)
    vg = VideoGenerator(opt)
    vg.gen_video()

if __name__=="__main__":
    main()