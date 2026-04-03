import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
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

def pir_to_lps(points):
    """
    Convert point(s) from PIR to LPS.
    points: shape (3,) or (N, 3) array-like in PIR order.
    Returns: same shape in LPS.
    """
    pts = np.asarray(points)
    M = np.array([[ 0,  0, -1],   # x_lps = -z_pir
                  [ 1,  0,  0],   # y_lps =  x_pir
                  [ 0, -1,  0]])  # z_lps = -y_pir
    if pts.ndim == 1:
        return M @ pts
    return pts @ M.T

def project_to_plane(verts, plane_axes, pitch=0.02, margin=0.5):
    """Project 3D mesh vertices to a 2D binary mask and return contours in mm."""
    X, Y = plane_axes
    pts = verts[:, [X, Y]]

    # Grid setup
    mins = pts.min(axis=0) - margin
    maxs = pts.max(axis=0) + margin
    res = np.ceil((maxs - mins) / pitch).astype(int)

    mask = np.zeros(res[::-1], dtype=bool)
    # convert coordinates to pixel indices
    ij = ((pts - mins) / pitch).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
            (ij[:, 0] < res[0]) & (ij[:, 1] < res[1])]
    mask[ij[:, 1], ij[:, 0]] = True
    mask = binary_dilation(mask, iterations=2)

    # extract contours and convert to mm coordinates
    contours = find_contours(mask.astype(float), 0.5)
    contour_mm = []
    for c in contours:
        x_mm = c[:, 1] * pitch + mins[0]
        y_mm = c[:, 0] * pitch + mins[1]
        contour_mm.append(np.column_stack((x_mm, y_mm)))
    return contour_mm