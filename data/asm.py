from pycpd import DeformableRegistration
import open3d as o3d
import os, glob
import copy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import plotly.graph_objects as go
import tqdm, time


def plot_pts(s1, s2):
    fig = go.Figure()
    for i in range(len(s1)):
        x = s1[i][:, 0].tolist()
        y = s1[i][:, 1].tolist()
        z = s1[i][:, 2].tolist()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=1),
                                   name='temp_{}'.format(i + 1)))

    for i in range(len(s2)):
        x = s2[i][:, 0].tolist()
        y = s2[i][:, 1].tolist()
        z = s2[i][:, 2].tolist()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=1),
                                   name='corr_temp_{}'.format(i + 1)))

    fig.show()


import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T


def shape_norm(pts, factor=10, return_para=False):
    pts_norm = copy.deepcopy(pts)
    center = pts.mean(0)
    pts_norm = pts_norm - center
    scale = np.sqrt((np.mean(np.sum(pts * pts, 1))))
    pts_norm *= (factor / scale)

    if return_para:
        return pts_norm, center, factor / scale
    else:
        return pts_norm


def align_shape(temp_pts, tgt_lst):
    n = temp_pts.shape[0]
    N = len(tgt_lst)
    tgts = np.zeros((N, n, 3))
    temp_pts_norm, temp_center, temp_scale = shape_norm(temp_pts, return_para=True)

    for i in tqdm.tqdm(range(N)):
        time.sleep(0.1)

        tgt_mesh = o3d.io.read_triangle_mesh(tgt_lst[i])
        tgt_pcd = tgt_mesh.sample_points_poisson_disk(n)
        tgt_pts = np.asarray(tgt_pcd.points)
        tgt_pts_norm, tgt_pts_center, tgt_pts_scale = shape_norm(tgt_pts, return_para=True)

        T = icp(temp_pts_norm, tgt_pts_norm)
        R = T[:3, :3]
        t = T[:3, 3]
        temp_pts_icp = temp_pts_norm.dot(R.T) + t.T
        reg = DeformableRegistration(X=tgt_pts_norm, Y=temp_pts_icp, alpha=1, beta=1)
        temp_pts_cpd, _ = reg.register()

        tgts[i] = temp_pts_cpd / tgt_pts_scale + tgt_pts_center

    return tgts


def npcvtobj(temp_mesh, pts):
    new_mesh = copy.deepcopy(temp_mesh)
    new_mesh.vertices = o3d.utility.Vector3dVector(pts)
    return new_mesh


def main():
    root = '/home/zyuaq/mesh/data/MSD/Heart'
    temp_mesh = o3d.io.read_triangle_mesh(os.path.join(root, 'temp.obj'))
    temp_pts = np.asarray(temp_mesh.vertices)

    tgt_lst = glob.glob(os.path.join(root, 'surfs_unaligned', '*.obj'))
    tgt_lst.sort()
    tgt_lst = tgt_lst[1:]
    temps_aligned_pts = align_shape(temp_pts, tgt_lst)
    align_shapes = np.concatenate([temp_pts[np.newaxis,], temps_aligned_pts], axis=0)

    for i in range(align_shapes.shape[0]):
        temps_aligned_mesh = npcvtobj(temp_mesh, align_shapes[i])
        o3d.io.write_triangle_mesh(os.path.join(root, 'surfs', '{:0>2d}surf.obj'.format(i + 1)),
                                   temps_aligned_mesh,
                                   write_vertex_normals=False,
                                   write_vertex_colors=False,
                                   write_triangle_uvs=False)


if __name__ == '__main__':
    main()
