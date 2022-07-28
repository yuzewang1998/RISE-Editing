import os
import torch
class BaseCheckpointsController:
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--editor_checkpoints_scans',
                            type=str,
                            default='edit_something',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='scans of checkpoints')
        parser.add_argument('--editor_checkpoints_root',
                            type=str,
                            default='/home/slam/devdata/NSEPN/checkpoints/col_nerfsynth',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets')
        return parser
    def __init__(self,opt,name=None):
        self.opt = opt
        if name is not None:
            self.checkpoints_name = name
        else:
            pth_files_dir = [f for f in os.listdir(os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans)) if f.endswith('_net_ray_marching.pth')]
            assert len(pth_files_dir)>0,'Empty checkpoints file'
            numlist = [int(i.split('_')[0]) for i in pth_files_dir]
            numlist.sort()
            latest_iters = str(numlist[-1])
            self.checkpoints_name = latest_iters +'_net_ray_marching'# find the latest pth file
        self.checkpoints_path = os.path.join(opt.editor_checkpoints_root,opt.editor_checkpoints_scans,self.checkpoints_name +'.pth')
        self.network_paras = torch.load(self.checkpoints_path, map_location=torch.device('cpu'))
    def cvt_2_neuralPoint(self):
        self.points_xyz = self.network_paras["neural_points.xyz"].view(-1, 3).cpu().numpy()
        self.points_embeding = self.network_paras["neural_points.points_embeding"].view(-1, 32).cpu().numpy()
        self.points_conf = self.network_paras["neural_points.points_conf"].view(-1, 1).cpu().numpy()
        self.points_color = self.network_paras["neural_points.points_color"].view(-1, 3).cpu().numpy()
    def set_and_save(self,points,name):
        raise NotImplementedError
