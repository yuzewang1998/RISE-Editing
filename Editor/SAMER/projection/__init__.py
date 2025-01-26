import numpy as np
import torch
import torch.nn.functional as F
import cv2
def find_project_function(name:str):
    if name == 'mask_projection':
        return mask_projection
    elif name == 'logits_projection':
        return logits_projection
    elif name == 'logits_uncertain_projection':
        return logits_uncertain_projection
    else:
        raise NotImplementedError
def mask_projection(uv, mask, thresh=0.5, depth=None, pnt_depth=None, depth_thresh=0.1):
    h, w = mask.shape
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    sample_mask = sample_mask > thresh
    if depth is not None:
        sample_depth = F.grid_sample(
            torch.from_numpy(depth).float()[None, None, ...], uv_rescale_ten,
            padding_mode="border", align_corners=True).reshape(-1).numpy()
        sample_mask = sample_mask & (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
    return sample_mask

def logits_projection(uv, mask, logits,mask_thresh=0.5, depth=None, pnt_depth=None, depth_thresh=0.1):
    h, w = mask.shape
    logits = cv2.resize(logits[0],(h,w))[None,...]
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    logits_ten = torch.from_numpy(logits).float()[None, ...]
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    sample_mask = sample_mask > mask_thresh

    sample_logits = F.grid_sample(
        logits_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    depth_mask = None
    if depth is not None:
        sample_depth = F.grid_sample(
            torch.from_numpy(depth).float()[None, None, ...], uv_rescale_ten,
            padding_mode="border", align_corners=True).reshape(-1).numpy()
        # sample_mask = sample_mask & (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
        depth_mask =  (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
    return sample_mask, sample_logits, depth_mask

def logits_projection(uv, mask, logits,mask_thresh=0.5, depth=None, pnt_depth=None, depth_thresh=0.1):
    h, w = mask.shape
    logits = cv2.resize(logits[0],(h,w))[None,...]
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    logits_ten = torch.from_numpy(logits).float()[None, ...]
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    sample_mask = sample_mask > mask_thresh

    sample_logits = F.grid_sample(
        logits_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    depth_mask = None
    if depth is not None:
        sample_depth = F.grid_sample(
            torch.from_numpy(depth).float()[None, None, ...], uv_rescale_ten,
            padding_mode="border", align_corners=True).reshape(-1).numpy()
        # sample_mask = sample_mask & (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
        depth_mask =  (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
    return sample_mask, sample_logits, depth_mask

# not a good method
def logits_uncertain_projection(uv, mask, logits,mask_thresh=0.5, depth=None, pnt_depth=None, depth_thresh=0.05,sigma = 0.1,depth_shift=0.0):
    h, w = mask.shape
    logits = cv2.resize(logits[0],(h,w))[None,...]
    h_w_half = np.array([w, h]) / 2
    uv_rescale = (uv[:, :2] - h_w_half) / h_w_half
    uv_rescale_ten = torch.from_numpy(uv_rescale).float()[None, None, ...]
    mask_ten = torch.from_numpy(mask).float()[None, None, ...]
    logits_ten = torch.from_numpy(logits).float()[None, ...]
    sample_mask = F.grid_sample(
        mask_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    sample_mask = sample_mask > mask_thresh

    sample_logits = F.grid_sample(
        logits_ten, uv_rescale_ten,
        padding_mode="border", align_corners=True).reshape(-1).numpy()
    depth_weight = None
    if depth is not None:
        sample_depth = F.grid_sample(
            torch.from_numpy(depth).float()[None, None, ...], uv_rescale_ten,
            padding_mode="border", align_corners=True).reshape(-1).numpy()
        dis = np.abs(sample_depth - pnt_depth[..., 0])
        dis = (dis)/np.var(dis)
        # sample_mask = sample_mask & (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
        # depth_mask =  (np.abs(sample_depth - pnt_depth[..., 0]) < depth_thresh)
        depth_weight = 1/((1/(2*np.pi*sigma*sigma)**(1/2))*np.exp((pnt_depth[...,0] - sample_depth-depth_shift)**2/(2*sigma*sigma))) # 0.1,0.2
        depth_weight = (1/depth_weight.max())*depth_weight
    return sample_mask, sample_logits, depth_weight


