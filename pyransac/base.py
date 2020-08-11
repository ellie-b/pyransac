"""Base definitions module.

This module contains definitions for base classes.
"""

# Standard library imports
import open3d as o3d

def make_transform(self, sample_ind, source, target) -> None:
    """Makes a model from given data points.

    :param sample_ind: indices for random sample to compute transformation
    :param source: SurfacePointCloud class from keypts.py
    :param target: SurfacePointCloud class from keypts.py
    """
    t_source = source.point_cloud.select_down_sample(sample_ind)
    t_target = target.point_cloud.select_down_sample(sample_ind)
    Tmatrix = o3d.registration.TransformationEstimationPointToPoint.compute_transformation(t_source,t_target,with_scaling = True)
    return Tmatrix

def calc_neighbors(self, src_mod,trg_pts) -> float:
    """Calculates nearest neighbors of target points in source points.

    :param src_mod: all points in source cloud after transformation SurfacePointCloud class from keypts.py
    :param trg_pts: initial matches for target cloud SurfacePointCloud class from keypts.py
    """
    
    tree = src_pts.kdtree
    trg_xyz = trg_pts.points
    
    [k, idx, _] = tree.search_knn_vector_3d(trg_xyz, 1)
    return idx