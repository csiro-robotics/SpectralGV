# Functions in this file are adapted from: https://github.com/ZhiChen902/SC2-PCR/blob/main/SC2_PCR.py

import numpy as np
import torch

def match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features):
    # normalize:
    src_features = torch.nn.functional.normalize(src_features, p=2.0, dim=1)
    tgt_features = torch.nn.functional.normalize(tgt_features, p=2.0, dim=1)

    distance = torch.cdist(src_features, tgt_features)
    min_vals, min_ids = torch.min(distance, dim=2)
 
    min_ids = min_ids.unsqueeze(-1).expand(-1, -1, 3)
    tgt_keypts_corr = torch.gather(tgt_keypts, 1, min_ids)
    src_keypts_corr = src_keypts

    return src_keypts_corr, tgt_keypts_corr

def power_iteration(M, num_iterations=5):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_pts, num_pts] the adjacency matrix
    Output:
        - leading_eig: [bs, num_pts] leading eigenvector
    """
    leading_eig = torch.ones_like(M[:, :, 0:1])
    leading_eig_last = leading_eig
    for i in range(num_iterations):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig


def cal_spatial_consistency( M, leading_eig):
    """
    Calculate the spatial consistency based on spectral analysis.
    Input:
        - M:          [bs, num_pts, num_pts] the adjacency matrix
        - leading_eig [bs, num_pts]           the leading eigenvector of matrix M
    Output:
        - sc_score_list [bs, 1]
    """
    spatial_consistency = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    spatial_consistency = spatial_consistency.squeeze(-1) / M.shape[1]
    return spatial_consistency


def sgv(src_keypts, tgt_keypts, src_features, tgt_features, d_thresh=5.0):
    """
    Input:
        - src_keypts: [1, num_pts, 3]
        - tgt_keypts: [bs, num_pts, 3]
        - src_features: [1, num_pts, D]
        - tgt_features: [bs, num_pts, D]
    Output:
        - sc_score_list:   [bs, 1], spatial consistency score for each candidate
    """
    # Correspondence Estimation: Nearest Neighbour Matching
    src_keypts_corr, tgt_keypts_corr = match_pair_parallel(src_keypts, tgt_keypts, src_features, tgt_features)

    # Spatial Consistency Adjacency Matrix
    src_dist = torch.norm((src_keypts_corr[:, :, None, :] - src_keypts_corr[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts_corr[:, :, None, :] - tgt_keypts_corr[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)
    adj_mat = torch.clamp(1.0 - cross_dist ** 2 / d_thresh ** 2, min=0)

    # Spatial Consistency Score
    lead_eigvec = power_iteration(adj_mat)
    sc_score_list = cal_spatial_consistency(adj_mat, lead_eigvec)

    sc_score_list = np.squeeze(sc_score_list.cpu().detach().numpy())
    return sc_score_list

def sgv_fn(query_keypoints, candidate_keypoints, d_thresh=5.0):

    kp1 = query_keypoints['keypoints']
    kp2 = candidate_keypoints['keypoints']
    f1 = query_keypoints['features']
    f2 = candidate_keypoints['features']
    
    # draw_registration_result(kp1, kp2, np.eye(4))

    min_num_feat = min(len(kp1),len(kp2))
    kp1 = kp1[:min_num_feat]
    kp2 = kp2[:min_num_feat]
    f1 = f1[:min_num_feat]
    f2 = f2[:min_num_feat]

    src_keypts = kp1.unsqueeze(0).cuda()
    tgt_keypts = kp2.unsqueeze(0).cuda() 
    src_features = f1.unsqueeze(0).cuda()
    tgt_features = f2.unsqueeze(0).cuda() 

    conf = sgv(src_keypts, tgt_keypts, src_features, tgt_features, d_thresh=d_thresh)



    return  conf