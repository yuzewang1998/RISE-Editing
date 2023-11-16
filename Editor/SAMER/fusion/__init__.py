import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('./fusion')

from neural_point_utils import *
def find_fusion_function(name:str):
    if name == 'simple_mask_logical_fusion':
        return simple_mask_logical_fusion
    elif name == 'depth_mask_logical_fusion':
        return depth_mask_logical_fusion
    elif name == 'simple_logits_logitsaverage_fusion':
        return simple_logits_logitsaverage_fusion
    elif name == 'depth_logits_logitsaverage_fusion':
        return depth_logits_logitsaverage_fusion
    elif name == 'simple_mask_voting_fusion':
        return simple_mask_voting_fusion
    elif name == 'depth_mask_voting_fusion':
        return depth_mask_voting_fusion
    elif name == 'depth_logits_distanceweightedaverage_fusion':
        return depth_logits_distanceweightedaverage_fusion
    elif name == 'udepth_logits_distanceweightedaverage_fusion':
        return udepth_logits_distanceweightedaverage_fusion


def simple_mask_logical_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map,**kwargs):
    pnt_frame_mask = project_func(uv_cam,mask)[...,None]
    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
    else:
        if kwargs['or_and_mode'] == 0 : #or
            pnt_mask_curr = np.logical_or(pnt_mask_curr, pnt_frame_mask)
        else:
            pnt_mask_curr = np.logical_and(pnt_mask_curr, pnt_frame_mask)
    return pnt_frame_buffer,pnt_mask_curr

def depth_mask_logical_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map, depth_pnt=None, depth_thresh=0.1, **kwargs):
    pnt_frame_mask = project_func(uv_cam, mask, 0.5, depth_map, depth_pnt, depth_thresh)[..., None]
    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
    else:
        if kwargs['or_and_mode'] == 0 : #or
            pnt_mask_curr = np.logical_or(pnt_mask_curr, pnt_frame_mask)
        else:
            pnt_mask_curr = np.logical_and(pnt_mask_curr, pnt_frame_mask)
    return pnt_frame_buffer,pnt_mask_curr

def depth_logits_logitsaverage_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map, logits = None, depth_pnt=None, depth_thresh=0.1, **kwargs):
    pnt_frame_mask, pnt_frame_logits, depth_mask = project_func(uv_cam, mask,logits ,0.5, depth_map, depth_pnt, depth_thresh)
    pnt_frame_mask = pnt_frame_mask[..., None]
    pnt_frame_logits = pnt_frame_logits[..., None]
    depth_mask = depth_mask[..., None]
    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'pnt_frame_logits':pnt_frame_logits,'depth_mask':depth_mask,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    '''
    pnt_frame_mask (222812, 1) bool
    pnt_frame_logits (222812, 1)
    depth_mask (222812,1)
    '''

    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
        pnt_mask_curr = pnt_mask_curr & depth_mask
    else:
        logits = np.array([data['pnt_frame_logits'] for data in pnt_frame_buffer])

        logits_avg = np.average(logits, axis = 0)
        pnt_depth_msk = torch.from_numpy(np.array([data['depth_mask'] for data in pnt_frame_buffer]))
        pnt_depth_msk = torch.any(pnt_depth_msk,dim=0,keepdim=True).numpy()
        # logits_avg_softmax = F.log_softmax(torch.from_numpy(logits_avg),dim=1).numpy()
        pnt_mask_curr = ((logits_avg > 0) &pnt_depth_msk)[0]
    return pnt_frame_buffer,pnt_mask_curr

def simple_logits_logitsaverage_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map,logits, **kwargs):
    pnt_frame_mask, pnt_frame_logits, _ = project_func(uv_cam,mask,logits)
    pnt_frame_mask = pnt_frame_mask[..., None]
    pnt_frame_logits = pnt_frame_logits[..., None]

    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'pnt_frame_logits':pnt_frame_logits,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    '''
    pnt_frame_mask (222812, 1) bool
    pnt_frame_logits (222812, 1)
    depth_mask (222812,1)
    '''

    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
    else:
        logits = np.array([data['pnt_frame_logits'] for data in pnt_frame_buffer])
        logits_avg = np.average(logits, axis = 0)
        # logits_avg_softmax = F.log_softmax(torch.from_numpy(logits_avg),dim=1).numpy()
        pnt_mask_curr = (logits_avg > 1)
    return pnt_frame_buffer,pnt_mask_curr

def simple_mask_voting_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map, **kwargs):
    pnt_frame_mask = project_func(uv_cam, mask)
    pnt_frame_mask = pnt_frame_mask[..., None]

    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    '''
    pnt_frame_mask (222812, 1) bool
    pnt_frame_logits (222812, 1)
    depth_mask (222812,1)
    '''

    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
    else:
        masks = np.array([data['pnt_frame_mask'] for data in pnt_frame_buffer])
        msk_vote = np.sum(masks, axis = 0,keepdims=True)

        pnt_mask_curr = (msk_vote > (masks.shape[0]/2))[0]
    return pnt_frame_buffer,pnt_mask_curr

def depth_mask_voting_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map, depth_pnt=None, depth_thresh=0.1, **kwargs):
    pnt_frame_mask = project_func(uv_cam, mask, 0.5, depth_map, depth_pnt, depth_thresh)
    pnt_frame_mask = pnt_frame_mask[..., None]
    pnt_frame_buffer.append(
        {'pnt_frame_mask': pnt_frame_mask, 'idx': image_idx, 'c2w': c2w, 'depth_map': depth_map, 'uv_cam': uv_cam})
    '''
    pnt_frame_mask (222812, 1) bool
    pnt_frame_logits (222812, 1)
    depth_mask (222812,1)
    '''
    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
    else:
        masks = np.array([data['pnt_frame_mask'] for data in pnt_frame_buffer])
        msk_vote = np.sum(masks, axis=0, keepdims=True)

        pnt_mask_curr = (msk_vote > (masks.shape[0] / 2))[0]
    return pnt_frame_buffer, pnt_mask_curr

def depth_logits_distanceweightedaverage_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map, logits = None, depth_pnt=None, depth_thresh=0.1,dis_func = "cosine", **kwargs):
    pnt_frame_mask, pnt_frame_logits, depth_mask = project_func(uv_cam, mask,logits ,0.5, depth_map, depth_pnt, depth_thresh)
    pnt_frame_mask = pnt_frame_mask[..., None]
    pnt_frame_logits = pnt_frame_logits[..., None]
    depth_mask = depth_mask[..., None]
    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'pnt_frame_logits':pnt_frame_logits,'depth_mask':depth_mask,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    '''
    pnt_frame_mask (222812, 1) bool
    pnt_frame_logits (222812, 1)
    depth_mask (222812,1)
    '''

    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
        pnt_mask_curr = pnt_mask_curr & depth_mask
    else:
        '''
        logits: [frame,pnt,1]
        logits_avg: [1,pnt,1]
        pnt_depth_msk: [frame, pnt, 1]
        pnt_depth_msk_and: [1,pnt,1]
        pnt_mask_curr [pnt, 1]
        '''
        logits = np.array([data['pnt_frame_logits'] for data in pnt_frame_buffer]) # [n, pnt, 1]

        logits_avg = np.average(logits, axis = 0,keepdims=True) # [1, pnt, 1]
        if dis_func == "cosine":
            weight = F.normalize(F.cosine_similarity(torch.from_numpy(logits),torch.from_numpy(logits_avg)),p=1,dim=0)[...,None]
        elif dis_func == "manhattan":
            weight = torch.sum(torch.abs(torch.from_numpy(logits) - torch.from_numpy(logits_avg)),dim=1)
            weight = F.softmax(-weight,dim=0)[:,None,:]
        elif dis_func == "euclidean":
            weight = F.normalize(torch.dist(torch.from_numpy(logits),torch.from_numpy(logits_avg)),p=2,dim=0)[...,None]
        else:
            raise NotImplementedError
        weighted_logits = torch.sum(torch.from_numpy(logits) * weight,dim = 0,keepdim=True)


        pnt_depth_msk = torch.from_numpy(np.array([data['depth_mask'] for data in pnt_frame_buffer]))
        pnt_depth_msk_and = torch.any(pnt_depth_msk,dim=0,keepdim=True).numpy()
        # logits_avg_softmax = F.log_softmax(torch.from_numpy(logits_avg),dim=1).numpy()
        pnt_mask_curr = np.logical_and((weighted_logits > 0), pnt_depth_msk_and)[0].numpy().astype(np.bool)
    return pnt_frame_buffer,pnt_mask_curr

def udepth_logits_distanceweightedaverage_fusion(pnt_mask_curr,uv_cam, mask, project_func,pnt_frame_buffer,image_idx,c2w,depth_map, logits = None, depth_pnt=None, depth_thresh=10e0,dis_func = "manhattan",weight_thresh = 0.1,**kwargs):
    pnt_frame_mask, pnt_frame_logits, depth_weight = project_func(uv_cam, mask,logits ,0.5, depth_map, depth_pnt, depth_thresh)
    pnt_frame_mask = pnt_frame_mask[..., None]
    pnt_frame_logits = pnt_frame_logits[..., None]
    depth_weight = depth_weight[..., None]
    pnt_frame_buffer.append({'pnt_frame_mask':pnt_frame_mask,'pnt_frame_logits':pnt_frame_logits,'depth_weight':depth_weight,'idx':image_idx, 'c2w':c2w, 'depth_map':depth_map, 'uv_cam':uv_cam})
    '''
    pnt_frame_mask (222812, 1) bool
    pnt_frame_logits (222812, 1)
    depth_mask (222812,1)
    '''

    if pnt_mask_curr is None:
        pnt_mask_curr = pnt_frame_mask.copy()
        pnt_mask_curr = (pnt_mask_curr*depth_weight)>weight_thresh
    else:
        '''
        logits: [frame,pnt,1]
        logits_avg: [1,pnt,1]
        pnt_depth_msk: [frame, pnt, 1]
        pnt_depth_msk_and: [1,pnt,1]
        pnt_mask_curr [pnt, 1]
        '''
        logits = np.array([data['pnt_frame_logits'] for data in pnt_frame_buffer]) # [n, pnt, 1]

        logits_avg = np.average(logits, axis = 0,keepdims=True) # [1, pnt, 1]
        if dis_func == "cosine":
            weight = F.normalize(F.cosine_similarity(torch.from_numpy(logits),torch.from_numpy(logits_avg)),p=1,dim=0)[...,None]
        elif dis_func == "manhattan":
            weight = torch.sum(torch.abs(torch.from_numpy(logits) - torch.from_numpy(logits_avg)),dim=1)
            weight = F.softmax(-weight,dim=0)[:,None,:]
        elif dis_func == "euclidean":
            weight = F.normalize(torch.dist(torch.from_numpy(logits),torch.from_numpy(logits_avg)),p=2,dim=0)[...,None]
        else:
            raise NotImplementedError
        weighted_logits = torch.sum(torch.from_numpy(logits) * weight,dim = 0,keepdim=True)


        pnt_depth_weight = np.array([data['depth_weight'] for data in pnt_frame_buffer])
        pnt_depth_weight = np.max(pnt_depth_weight,axis=0)[None,...]

        # logits_avg_softmax = F.log_softmax(torch.from_numpy(logits_avg),dim=1).numpy()
        pnt_mask_curr = ((weighted_logits * pnt_depth_weight)>weight_thresh)[0].numpy()
    return pnt_frame_buffer,pnt_mask_curr

