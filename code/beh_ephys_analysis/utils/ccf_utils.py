import numpy as np
# ccf_res = 25
# bregma_points = np.array([216, 18, 228])  # Example bregma points in ccf space
# voxel_size = 25

def ccf_pts_convert_to_mm(ccf_pts, bregma_points=None, ccf_res=None):
    if bregma_points is None:
        bregma_points = np.array([216, 18, 228]) # PIR bregma points in ccf space
    if ccf_res is None:
        ccf_res = 25
    ccf_pts_mm = (ccf_pts - bregma_points) * ccf_res / 1000  # Convert to mm
    if np.size(ccf_pts_mm,0) == 1:
        ccf_pts_mm[0] = -1 * ccf_pts_mm[0]  # flip AP-axis
    else:
        ccf_pts_mm[:, 0] = -1 * ccf_pts_mm[:, 0]  # flip AP-axis
    return ccf_pts_mm