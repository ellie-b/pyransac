"""Random sample consensus (RANSAC) module.

This module contains the core code for the RANSAC algorithm.

Originally written by Adam Morrissett

Adapted by Eleanor Bash to filter 3D point cloud correspondences bsaed on 

Persad, RA, and C. Armenakis (2017). Automatic 3D Surface Co-Registration 
Using Keypoint Matching Photogrammetric Engineering & Remote Sensing, 83(2): 
    137-151.
"""

# Standard library imports
from dataclasses import dataclass
from math import log
import random

# Local application imports
import base

@dataclass
class RansacParams:
    """Random sample consensus (RANSAC) function parameters.

    This class contains the parameters for the RANSAC algorithm.
    """
    samples: 3
    """The number of random samples to take per iteration."""

    iterations: int
    """Maximum number iterations to complete."""

    # confidence: float
    # """The RANSAC confidence value (0 <= confidence <= 1)."""

    # threshold: float
    # """The error threshold to consider a point an inlier"""


def find_inliers(source,target,src_ind,trg_ind, params: RansacParams):
    """Find the inliers from a data set.

    Finds the inliers from a given data set given a model and
    an error function.

    :param target: data points in desired reference frame SurfacePointCloud class from keypts.py
    :param source: data points in reference frame to be transformed SurfacePointCloud class from keypts.py
    :param src_ind: indices from cost matrix for source points
    :param trg_ind: indices from cost matrix for target points
    :param params: parameters for the RANSAC algorithm
    :return: inliers, number of supporting points
    """
    par_idx = numpy.arange(0,len(ptCloud.xyz))
    inliers = []
    max_support = 0
    iterations = params.iterations
    i = 0

    while i < iterations:
        #adjust to select indices rather than points
        sample_ind = random.choices(par_idx, k=params.samples)
        
        #find transform for this iteration
        M = base.make_transform(sample_ind,source,target)
        
        #transform source points
        model = source.transform_pts(M)
        corres_t = target.point_cloud.select_down_sample(trg_ind)
        supporters = find_supporters(corres_t,model,src_ind)

        if len(supporters) > max_support:
            max_support = len(supporters)
            inliers = supporters

            confidence = 1 - params.confidence
            ratio = len(supporters) / len(trg_ind)

            # We cannot get more support than all data points
            if ratio == 1:
                break

            iterations = log(confidence) / log(1 - ratio ** params.samples)

        i += 1

    return inliers, max_support


def find_supporters(corres_t, model, src_ind):
    """Find data points (supporters) that support the given transformation.

    :param cores_t: data points target point cloud correspondences
    :param model: source point cloud transformed
    :param src_ind: indices for points which are initial matches for corres_t
    :return: data points that support the hypothesis
    """
    neighbors = base.calc_neighbors(model, corres_t) #indices of NN to each intial target cloud match
    
    return [point for point in neighbors if neighbors[point] == src_ind[point]]
