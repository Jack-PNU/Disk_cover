# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:45:35 2023

@author: GAO
"""

import numpy as np
import pandas as pd
from time import time
from scipy.spatial import distance

def is_approx_equal(sub_array, flat_array):
    return len(sub_array) == len(flat_array) and np.allclose(sub_array, flat_array, atol=1e-3)

class ProjectorStack:
    """
    Stack of points that are shifted / projected to put first one at origin.
    """
    def __init__(self, vec):
        self.vs = np.array(vec)
        
    def push(self, v):
        if len(self.vs) == 0:
            self.vs = np.array([v])
        else:
            self.vs = np.append(self.vs, [v], axis=0)
        return self
    
    def pop(self):
        if len(self.vs) > 0:
            ret, self.vs = self.vs[-1], self.vs[:-1]
            return ret
    
    def __mul__(self, v):
        s = np.zeros(len(v))
        for vi in self.vs:
            s = s + vi * np.dot(vi, v)
        return s
    
class GaertnerBoundary:
    """
        GärtnerBoundary

    See the passage regarding M_B in Section 4 of Gärtner's paper.
    """
    def __init__(self, pts):
        self.projector = ProjectorStack([])
        self.centers, self.square_radii = np.array([]), np.array([])
        self.empty_center = np.array([np.NaN for _ in pts[0]])


def push_if_stable(bound, pt):
    if len(bound.centers) == 0:
        bound.square_radii = np.append(bound.square_radii, 0.0)
        bound.centers = np.array([pt])
        return True
    q0, center = bound.centers[0], bound.centers[-1]
    C, r2  = center - q0, bound.square_radii[-1]
    Qm, M = pt - q0, bound.projector
    Qm_bar = M * Qm
    residue, e = Qm - Qm_bar, sqdist(Qm, C) - r2
    z, tol = 2 * sqnorm(residue), np.finfo(float).eps * max(r2, 1.0)
    isstable = np.abs(z) > tol
    if isstable:
        center_new  = center + (e / z) * residue
        r2new = r2 + (e * e) / (2 * z)
        bound.projector.push(residue / np.linalg.norm(residue))
        bound.centers = np.append(bound.centers, np.array([center_new]), axis=0)
        bound.square_radii = np.append(bound.square_radii, r2new)
    return isstable

def pop(bound):
    n = len(bound.centers)
    bound.centers = bound.centers[:-1]
    bound.square_radii = bound.square_radii[:-1]
    if n >= 2:
        bound.projector.pop()
    return bound


class NSphere:
    def __init__(self, c, sqr):
        self.center = np.array(c)
        self.sqradius = sqr

def isinside(pt, nsphere, atol=1e-6, rtol=0.0):
    r2, R2 = sqdist(pt, nsphere.center), nsphere.sqradius
    return r2 <= R2 or np.isclose(r2, R2, atol=atol**2,rtol=rtol**2)

def allinside(pts, nsphere, atol=1e-6, rtol=0.0):
    for p in pts:
        if not isinside(p, nsphere, atol, rtol):
            return False
    return True

def move_to_front(pts, i):
    pt = pts[i]
    for j in range(len(pts)):
        pts[j], pt = pt, np.array(pts[j])
        if j == i:
            break
    return pts

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def sqdist(p1, p2):
    return sqnorm(p1 - p2)

def sqnorm(p):
    return np.sum(np.array([x * x for x in p]))

def ismaxlength(bound):
    len(bound.centers) == len(bound.empty_center) + 1

def makeNSphere(bound):
    if len(bound.centers) == 0: 
        return NSphere(bound.empty_center, 0.0)
    return NSphere(bound.centers[-1], bound.square_radii[-1])

def _welzl(pts, pos, bdry):
    support_count, nsphere = 0, makeNSphere(bdry)
    if ismaxlength(bdry):
        return nsphere, 0
    for i in range(pos):
        if not isinside(pts[i], nsphere):
            isstable = push_if_stable(bdry, pts[i])
            if isstable:
                nsphere, s = _welzl(pts, i, bdry)
                pop(bdry)
                move_to_front(pts, i)
                support_count = s + 1
    return nsphere, support_count

def find_max_excess(nsphere, pts, k1):
    err_max, k_max = -np.Inf, k1 - 1
    for (k, pt) in enumerate(pts[k_max:]):
        err = sqdist(pt, nsphere.center) - nsphere.sqradius
        if  err > err_max:
            err_max, k_max = err, k + k1
    return err_max, k_max - 1

#最小圆的圆心在平面的任意位置
def welzl(points, maxiterations=2000):
    pts, eps = np.array(points, copy=True), np.finfo(float).eps
    bdry, t = GaertnerBoundary(pts), 1
    nsphere, s = _welzl(pts, t, bdry)
    for i in range(maxiterations):
        e, k = find_max_excess(nsphere, pts, t + 1)
        if e <= eps:
            break
        pt = pts[k]
        push_if_stable(bdry, pt)
        nsphere_new, s_new = _welzl(pts, s, bdry)
        pop(bdry)
        move_to_front(pts, k)
        nsphere = nsphere_new
        t, s = s + 1, s_new + 1
    return nsphere

def initialize_centers(data, R):
    # 随机选择第一个中心
    centers = [data[np.random.randint(data.shape[0])]]
    
    while True:
        # 计算每个数据点到最近中心的距离 distances
        #distances = np.min(np.linalg.norm(data - centers[:, np.newaxis], axis=2), axis=0)
        distances_matrix = distance.cdist(data, centers)
        distances = np.min(distances_matrix, axis=1)
        
        # 选择下一个中心，如果距离最大的点的距离大于 R
        next_center = np.argmax(distances)

        if distances[next_center] > R:
            centers.append(data[next_center])
        else:
            break
    #distances = np.min(np.linalg.norm(data - centers[:, np.newaxis], axis=2), axis=0)
    distances = np.min(distance.cdist(data, centers), axis=1)
    next_center = np.argmax(distances)
    centers.append(data[next_center])

    return centers

def init_upperbound(data, R, num_populations):
    
    # 第一次初始中心作为upperbound
    best_centers = initialize_centers(data, R)
    upperbound_k = len(best_centers)

    #在重复初始值选择最佳的一个
    for i in range(num_populations-1):
        centers = initialize_centers(data, R)
        k = len(centers)
        
        if k < upperbound_k:
            best_centers = centers
            upperbound_k = k
            
    # Calculate distances between each data point and all centers
    #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
    distances_matrix = distance.cdist(data, best_centers)
    distances = np.min(distances_matrix, axis=1)
    # Find the index of the center to which each data point is assigned
    data_indices = np.argmin(distances_matrix, axis=1)
    
    return upperbound_k, data_indices, distances#max(distances)
    
def circumcircle(p_1, p_2, p_3):
    """
    :return:  x0 and y0 is center of a circle, r is radius of a circle
    """
    x1, y1 = p_1
    x2, y2 = p_2
    x3, y3 = p_3
    x0 = 1/2*((x1**2+y1**2)*(y3-y2)+(x2**2+y2**2)*(y1-y3)+(x3**2+y3**2)*(y2-y1))/(x1*(y3-y2)+x2*(y1-y3)+x3*(y2-y1))
    y0 = 1/2*((x1**2+y1**2)*(x3-x2)+(x2**2+y2**2)*(x1-x3)+(x3**2+y3**2)*(x2-x1))/(y1*(x3-x2)+y2*(x1-x3)+y3*(x2-x1))
    center = np.array([x0,y0])
    radius = distance.euclidean(center, p_1)
    return center, radius

def get_minimum_enclosing_circle(points):
    # Step 1
    distances = distance.cdist(points, points)
    max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
    p1 = points[max_distance_idx[0]]
    p2 = points[max_distance_idx[1]]
    radius = distance.euclidean(p1, p2) / 2
    center = (p1 + p2) / 2

    # Check if all points are covered
    if np.all(np.round(distance.cdist([center], points),5) <= np.round(radius,5)):
        return center

    # Step 2 and Step 3
    else:
        center_distances = distance.cdist([center], points)
        max_cendis_idx = np.unravel_index(np.argmax(center_distances), center_distances.shape)
        p3 = points[max_cendis_idx[1]]
        center, radius = circumcircle(p1, p2, p3)

        # Check if all points are covered
        #print(distance.cdist([center], points),radius)
        if np.all(np.round(distance.cdist([center], points),5) <= np.round(radius,5)):
            return center
        
        else:
            center_distances = distance.cdist([center], points)
            max_cendis_idx = np.unravel_index(np.argmax(center_distances), center_distances.shape)
            p4 = points[max_cendis_idx[1]]
            
            #print(p1,p2,p3,p4)
            # Step 4
            d1 = distance.euclidean(p1, p3)
            d2 = distance.euclidean(p1, p4)
            d3 = distance.euclidean(p2, p3)
            d4 = distance.euclidean(p2, p4)
            
            # Choose the appropriate point to omit
            if min(d1, d2, d3, d4) == d1 or min(d1, d2, d3, d4) == d2:
                center, radius = circumcircle(p2, p3, p4)
                return center
            else:
                center, radius = circumcircle(p1, p3, p4)
                return center
            
def update_centers(data, num_clusters, cluster_index):
    
    #cluster_index = np.argmin(np.linalg.norm(data[:, np.newaxis] - circle_center, axis=2), axis=1)
    cluster_centers = []
    for idx in range(num_clusters):
        clust_idx = np.where(cluster_index == idx)[0]
        if len(clust_idx) == 0:
            continue
        elif len(clust_idx) == 1:
            center = data[clust_idx][0]
        elif len(clust_idx) == 2:
            #center, radius = two_point_circle(data[clust_idx])
            center = (data[clust_idx][0]+data[clust_idx][1])/2
        else:
            one_cluster_data = data[clust_idx]
            #一个最小圆的圆心
            #nsphere = welzl(one_cluster_data)
            #center = nsphere.center
            center = get_minimum_enclosing_circle(one_cluster_data)
        cluster_centers.append(center)
        
    return np.array(cluster_centers)

#这个是移除一个中心的函数，在这里没有用到
def remove_center(data, centers, R, removal_strategy):
    if removal_strategy == 'random':
        # 随机删除一个中心
        # 计算每个数据点到最近中心的距离
        #distances = np.min(np.linalg.norm(data - centers[:, np.newaxis], axis=2), axis=0)
        distances = np.min(distance.cdist(data, centers), axis=1)
        
        # 选择下一个中心，如果距离最大的点的距离大于 R
        if max(distances) <= R:
            removed_center = np.random.choice(len(centers))
        
    elif removal_strategy == 'D':
        # 删除 D 所在的中心
        #distances that each data point to all centers
        #distances_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        distances_matrix = distance.cdist(data, centers)
        distances = np.min(distances_matrix, axis=1)

        # 选择离最近中心的最远距离的数据点的索引 [0.3 0.7  0.36055513 0.72801099]
        bigdisc_data = np.argmax(distances)
        
        # 将每个数据点分配给最近中心的索引 [0 0 1 1]
        assigned_indices = np.argmin(distances, axis=1)
        
        removed_center = assigned_indices[bigdisc_data]  #删除中心的index
        
    elif removal_strategy == 'S':
        # 删除 S 所在的中心
        #distances = np.max(np.linalg.norm(data - np.array(centers)[:, np.newaxis], axis=2), axis=0)
        # Calculate distances between each data point and all centers
        distances = distance.cdist(data, centers)
        # Find the index of the center to which each data point is assigned
        assigned_indices = np.argmin(distances, axis=1)
        
        # Find the maximum distance from each center to the data points assigned to it
        max_distances = np.array([np.max(distances[assigned_indices == i, i]) for i in range(len(centers))])
        # Find the minimum of the maximum distances
        removed_center = np.argmin(max_distances)  #删除中心的index
        
    else:
        raise ValueError("Invalid removal strategy")
    
    # 删除中心
    del centers[removed_center]

    return centers

def DR_dist(data, R, num_populations, removal_strategy):
    
    K, data_indices, distances = init_upperbound(data, R, num_populations)
    radii = distances
    itera = 0
    
    while True:
        centers = update_centers(data, K, data_indices)

        #distances that each data point to all centers
        #distances_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        distances_matrix = distance.cdist(data, centers)
        distances = np.min(distances_matrix, axis=1)
        if max(distances) <= R:
            if removal_strategy == 'random':
                removed_center = np.random.choice(len(centers))
                
            elif removal_strategy == 'max_circle':
                # 选择离最近中心的最远距离的数据点的索引 [0.3 0.7  0.36055513 0.72801099]
                bigdisc_data = np.argmax(distances)  # 3
                # 将每个数据点分配给最近中心的索引 [0 0 1 1]
                assigned_indices = np.argmin(distances_matrix, axis=1)
                removed_center = assigned_indices[bigdisc_data]  #删除中心的index
            
            elif removal_strategy == 'min_circle':
                # Find the index of the center to which each data point is assigned
                assigned_indices = np.argmin(distances_matrix, axis=1)
                
                # Find the maximum distance from each center to the data points assigned to it
                max_distances = np.array([np.max(distances_matrix[assigned_indices == i, i]) for i in range(len(centers))])
                # Find the minimum of the maximum distances
                removed_center = np.argmin(max_distances)  #删除中心的index
                
            else:
                raise ValueError("Invalid removal strategy")
            
            # 删除中心 del centers[removed_center]
            best_centers = centers
            centers = np.delete(centers, removed_center, axis=0)
            #最大距离必须大于给定的半径
            best_K = K
            K -= 1
            #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
            distances_matrix = distance.cdist(data, centers)
            # Find the index of the center to which each data point is assigned
            data_indices = np.argmin(distances_matrix, axis=1)
            
        else:
            print(distances, max(distances))
            #if max(distances) == max_distance:
            if np.array_equal(radii, distances): #np.allclose(radii, distances, rtol=1e-01, atol=1e-03): #(radii == distances).all(): 
                return best_K, best_centers
            else:
                #max_distance = max(distances)
                #diff_elements = np.setdiff1d(radii, distances) #检测两个数组中不同的元素
                #print(len(diff_elements))
                itera += 1
                if itera % 2 == 0:
                    radii = distances
                data_indices = np.argmin(distances_matrix, axis=1)
                
#用需求点到最近中心的最远距离比较        
def DR_mix(data, R, num_populations, removal_strategy):
    
    K, data_indices, distances = init_upperbound(data, R, num_populations)
    max_distance = max(distances)
    k_radii = []
    
    while True:
        centers = update_centers(data, K, data_indices)
        
        #distances that each data point to all centers
        #distances_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        distances_matrix = distance.cdist(data, centers)
        distances = np.min(distances_matrix, axis=1)
        if max(distances) <= R:
            if removal_strategy == 'random':
                removed_center = np.random.choice(len(centers))
                
            elif removal_strategy == 'max_circle':
                # 选择离最近中心的最远距离的数据点的索引 [0.3 0.7  0.36055513 0.72801099]
                bigdisc_data = np.argmax(distances)  # 3
                # 将每个数据点分配给最近中心的索引 [0 0 1 1]
                assigned_indices = np.argmin(distances_matrix, axis=1)
                removed_center = assigned_indices[bigdisc_data]  #删除中心的index
            
            elif removal_strategy == 'min_circle':
                # Find the index of the center to which each data point is assigned
                assigned_indices = np.argmin(distances_matrix, axis=1)
                
                # Find the maximum distance from each center to the data points assigned to it
                max_distances = np.array([np.max(distances_matrix[assigned_indices == i, i]) for i in range(len(centers))])
                # Find the minimum of the maximum distances
                removed_center = np.argmin(max_distances)  #删除中心的index
                
            else:
                raise ValueError("Invalid removal strategy")
            
            # 删除中心 del centers[removed_center]
            best_centers = centers
            centers = np.delete(centers, removed_center, axis=0)
            #最大距离必须大于给定的半径
            best_K = K
            K -= 1
            mark = 0
            #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
            distances_matrix = distance.cdist(data, centers)
            # Find the index of the center to which each data point is assigned
            data_indices = np.argmin(distances_matrix, axis=1)
            
        else:
            #print(distances, max(distances))
            if mark == 0:
                if max(distances) == max_distance:
                    k_radii.append(distances)
                    data_indices = np.argmin(distances_matrix, axis=1)
                    mark = 1
                    
                else:
                    max_distance = max(distances)
                    data_indices = np.argmin(distances_matrix, axis=1)
                    
            elif mark == 1:
                #if any(np.array_equal(distances, arr) for arr in k_radii):
                if any(is_approx_equal(arr, distances) for arr in k_radii):
                    return best_K, best_centers
                else:
                    k_radii.append(distances)
                    data_indices = np.argmin(distances_matrix, axis=1)
                    
def DR(data, R, num_populations, removal_strategy):
    
    K, data_indices, distances = init_upperbound(data, R, num_populations)
    #max_distance = max(distances)
    k_radii = []
    while True:
        centers = update_centers(data, K, data_indices)
        
        #distances that each data point to all centers
        #distances_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        distances_matrix = distance.cdist(data, centers)
        distances = np.min(distances_matrix, axis=1)
        if max(distances) <= R:
            if removal_strategy == 'random':
                removed_center = np.random.choice(len(centers))
                
            elif removal_strategy == 'max_circle':
                # 选择离最近中心的最远距离的数据点的索引 [0.3 0.7  0.36055513 0.72801099]
                bigdisc_data = np.argmax(distances)  # 3
                # 将每个数据点分配给最近中心的索引 [0 0 1 1]
                assigned_indices = np.argmin(distances_matrix, axis=1)
                removed_center = assigned_indices[bigdisc_data]  #删除中心的index
            
            elif removal_strategy == 'min_circle':
                # Find the index of the center to which each data point is assigned
                assigned_indices = np.argmin(distances_matrix, axis=1)
                
                # Find the maximum distance from each center to the data points assigned to it
                max_distances = np.array([np.max(distances_matrix[assigned_indices == i, i]) for i in range(len(centers))])
                # Find the minimum of the maximum distances
                removed_center = np.argmin(max_distances)  #删除中心的index
                
            else:
                raise ValueError("Invalid removal strategy")
            
            # 删除中心 del centers[removed_center]
            best_centers = centers
            centers = np.delete(centers, removed_center, axis=0)
            #最大距离必须大于给定的半径
            best_K = K
            K -= 1
            k_radii = []
            #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
            distances_matrix = distance.cdist(data, centers)
            # Find the index of the center to which each data point is assigned
            data_indices = np.argmin(distances_matrix, axis=1)
            
        else:
            #print(distances, max(distances))
            if any(np.array_equal(distances, arr) for arr in k_radii):
            #if any(is_approx_equal(arr, distances) for arr in k_radii):
                return best_K, best_centers
            else:
                k_radii.append(distances)
                data_indices = np.argmin(distances_matrix, axis=1)
                
                    
def init_upperbound_1(data, R, num_populations):
    
    # 第一次初始中心作为upperbound
    best_centers = initialize_centers(data, R)
    upperbound_k = len(best_centers)

    #在重复初始值选择最佳的一个
    for i in range(num_populations-1):
        centers = initialize_centers(data, R)
        k = len(centers)
        
        if k < upperbound_k:
            best_centers = centers
            upperbound_k = k
            
    # Calculate distances between each data point and all centers
    #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
    distances_matrix = distance.cdist(data, best_centers)
    #distances = np.min(distances_matrix, axis=1)
    # Find the index of the center to which each data point is assigned
    data_indices = np.argmin(distances_matrix, axis=1)
    
    return upperbound_k, data_indices#distances#max(distances)
                                       
def get_minimum_enclosing_circle_1(points):
    # Step 1
    distances = distance.cdist(points, points)
    max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
    p1 = points[max_distance_idx[0]]
    p2 = points[max_distance_idx[1]]
    radius = distance.euclidean(p1, p2) / 2
    center = (p1 + p2) / 2

    # Check if all points are covered
    if np.all(np.round(distance.cdist([center], points),5) <= np.round(radius,5)):
        return center, radius

    # Step 2 and Step 3
    else:
        center_distances = distance.cdist([center], points)
        max_cendis_idx = np.unravel_index(np.argmax(center_distances), center_distances.shape)
        p3 = points[max_cendis_idx[1]]
        center, radius = circumcircle(p1, p2, p3)

        # Check if all points are covered
        #print(distance.cdist([center], points),radius)
        if np.all(np.round(distance.cdist([center], points),5) <= np.round(radius,5)):
            return center, radius
        
        else:
            center_distances = distance.cdist([center], points)
            max_cendis_idx = np.unravel_index(np.argmax(center_distances), center_distances.shape)
            p4 = points[max_cendis_idx[1]]
            
            #print(p1,p2,p3,p4)
            # Step 4
            d1 = distance.euclidean(p1, p3)
            d2 = distance.euclidean(p1, p4)
            d3 = distance.euclidean(p2, p3)
            d4 = distance.euclidean(p2, p4)
            
            # Choose the appropriate point to omit
            if min(d1, d2, d3, d4) == d1 or min(d1, d2, d3, d4) == d2:
                center, radius = circumcircle(p2, p3, p4)
                return center, radius
            else:
                center, radius = circumcircle(p1, p3, p4)
                return center, radius

def update_centers_1(data, num_clusters, cluster_index):
    
    #cluster_index = np.argmin(np.linalg.norm(data[:, np.newaxis] - circle_center, axis=2), axis=1)
    cluster_centers = []
    radii = []
    for idx in range(num_clusters):
        clust_idx = np.where(cluster_index == idx)[0]
        if len(clust_idx) == 0:
            continue
        elif len(clust_idx) == 1:
            center = data[clust_idx][0]
            radius = 0
        else:
            one_cluster_data = data[clust_idx]
            #一个最小圆的圆心
            #nsphere = welzl(one_cluster_data)
            #center = nsphere.center
            center, radius = get_minimum_enclosing_circle_1(one_cluster_data)
        cluster_centers.append(center)
        radii.append(radius)
    return np.array(cluster_centers), np.array(radii)

#用簇形成的最大圆的半径       
def DR_dist_1(data, R, num_populations):
    mark = 1
    K, data_indices = init_upperbound_1(data, R, num_populations)
    """
    if K < 10:
        removal_strategy = 'max_circle'
    elif K>=10 and K<15:
        removal_strategy = 'random'
    else:
        removal_strategy = 'min_circle'
    """ 
    max_distance = float('inf')
    k_radii = []
    while True:
        centers, radii = update_centers_1(data, K, data_indices)
        #distances that each data point to all centers
        #distances_matrix = distance.cdist(data, centers)
        #distances = np.min(distances_matrix, axis=1)
        print(radii)
        if max(radii) <= R:
            if K>13:
                removed_center = np.argmin(radii)
                
            elif K>7 and K<14:
                removed_center = np.random.choice(len(centers))#删除中心的index
            
            elif K<8:
                removed_center = np.argmax(radii)  #删除中心的index
                
            else:
                raise ValueError("Invalid removal strategy")
            
            # 删除中心 del centers[removed_center]
            best_centers = centers
            centers = np.delete(centers, removed_center, axis=0)
            #最大距离必须大于给定的半径
            best_K = K
            K -= 1
            mark = 0
            #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
            distances_matrix = distance.cdist(data, centers)
            # Find the index of the center to which each data point is assigned
            data_indices = np.argmin(distances_matrix, axis=1)
            
        else:
            #print(radii, max(radii))
            if mark == 0:
                if max(radii) == max_distance:
                    k_radii.append(radii)
                    distances_matrix = distance.cdist(data, centers)
                    data_indices = np.argmin(distances_matrix, axis=1)
                    mark = 1
                    
                else:
                    max_distance = max(radii)
                    distances_matrix = distance.cdist(data, centers)
                    data_indices = np.argmin(distances_matrix, axis=1)
                    
            elif mark == 1:
                #if any(np.array_equal(radii, arr) for arr in k_radii):
                print(any(is_approx_equal(arr, radii) for arr in k_radii))
                if any(is_approx_equal(arr, radii) for arr in k_radii):
                    return best_K, best_centers
                else:
                    k_radii.append(radii)
                    distances_matrix = distance.cdist(data, centers)
                    data_indices = np.argmin(distances_matrix, axis=1)
                    
def DR_1(data, R, num_populations):

    K, data_indices = init_upperbound_1(data, R, num_populations)
    #print(K)

    if K > 0:
        removal_strategy = 'min_circle'
    #elif K>=10 and K<15:
        #removal_strategy = 'random'
    else:
        removal_strategy = 'max_circle'

    k_radii = []
    while True:
        centers, radii = update_centers_1(data, K, data_indices)
        #distances that each data point to all centers
        #distances_matrix = distance.cdist(data, centers)
        #distances = np.min(distances_matrix, axis=1)
        #print(radii)
        if max(radii) <= R:
            #if K>0:
            if removal_strategy == 'min_circle':
                removed_center = np.argmin(radii)
            #removed_center = np.random.choice(len(centers))    
            #elif K>5 and K<9:
                #removed_center = np.random.choice(len(centers))#删除中心的index
            
            else:
                removed_center = np.argmax(radii)  #删除中心的index
                
            #else:
                #raise ValueError("Invalid removal strategy")
                
            k_radii = []
            # 删除中心 del centers[removed_center]
            best_centers = centers
            centers = np.delete(centers, removed_center, axis=0)
            #最大距离必须大于给定的半径
            best_K = K
            K -= 1
            
            best_data_indices = data_indices  #得到的各个数据属于各中心的索引
            centerindex_allocation = np.argmax(radii)  #需要重新分配的中心
            
            #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
            distances_matrix = distance.cdist(data, centers)
            # Find the index of the center to which each data point is assigned
            data_indices = np.argmin(distances_matrix, axis=1)
            
        else:
            #print(radii, max(radii))
            if any(np.array_equal(radii, arr) for arr in k_radii):
            #if any(is_approx_equal(arr, radii) for arr in k_radii):
                return best_K, best_centers, best_data_indices, centerindex_allocation
            else:
                k_radii.append(radii)
                distances_matrix = distance.cdist(data, centers)
                data_indices = np.argmin(distances_matrix, axis=1)

def DR_2(data, R, K, data_indices):
    
    if K > 0:
        removal_strategy = 'min_circle'
    #elif K>=10 and K<15:
        #removal_strategy = 'random'
    else:
        removal_strategy = 'max_circle'

    k_radii = []
    centers, radii = update_centers_1(data, K, data_indices)
    best_K = K
    best_centers = centers
    best_data_indices = data_indices
    centerindex_allocation = np.argmax(radii)
    
    while True:
        centers, radii = update_centers_1(data, K, data_indices)
        #distances that each data point to all centers
        #distances_matrix = distance.cdist(data, centers)
        #distances = np.min(distances_matrix, axis=1)
        #print(radii)
        if max(radii) <= R:
            #if K>0:
            if removal_strategy == 'min_circle':
                removed_center = np.argmin(radii)
            #removed_center = np.random.choice(len(centers))    
            #elif K>5 and K<9:
                #removed_center = np.random.choice(len(centers))#删除中心的index
            
            else:
                removed_center = np.argmax(radii)  #删除中心的index
                
            #else:
                #raise ValueError("Invalid removal strategy")
                
            k_radii = []
            # 删除中心 del centers[removed_center]
            best_centers = centers
            centers = np.delete(centers, removed_center, axis=0)
            #最大距离必须大于给定的半径
            best_K = K
            K -= 1
            
            best_data_indices = data_indices  #得到的各个数据属于各中心的索引
            centerindex_allocation = np.argmax(radii)  #需要重新分配的中心
            
            #distances_matrix = np.linalg.norm(data[:, np.newaxis] - best_centers, axis=2)
            distances_matrix = distance.cdist(data, centers)
            # Find the index of the center to which each data point is assigned
            data_indices = np.argmin(distances_matrix, axis=1)
            
        else:
            #print(radii, max(radii))
            if any(np.array_equal(radii, arr) for arr in k_radii):
            #if any(is_approx_equal(arr, radii) for arr in k_radii):
                return best_K, best_centers, best_data_indices, centerindex_allocation
            else:
                k_radii.append(radii)
                distances_matrix = distance.cdist(data, centers)
                data_indices = np.argmin(distances_matrix, axis=1)

def Enhanced_DR(data, R, K, centers, data_labels, center_change):
    
    while True:
        cluster_points = data[data_labels == center_change]
        if len(cluster_points) > 1:
            # 随机选择该簇中的一个点作为新的中心
            new_center = cluster_points[np.random.choice(len(cluster_points))]
            centers[center_change] = new_center
        distances_matrix = distance.cdist(data, centers)
        # Find the index of the center to which each data point is assigned
        data_indices = np.argmin(distances_matrix, axis=1)
        number_disk, center_locations, cluster_labels, centerindex_allocation = DR_2(data, R, K, data_indices)
        if number_disk < K:
            K = number_disk
            centers = center_locations
            data_labels = cluster_labels
            center_change = centerindex_allocation
        else:
            return K#, centers, data_labels, center_change
               
if __name__ == "__main__":
    
    R = 500
    num_populations = 1
    #removal_strategy = 'min_circle'
    
    file_path = r"3038.txt"  #1291,1889,2319, 3038
    data_raw = np.loadtxt(file_path, delimiter = " ")
    
    k_random = []
    time_random = []
    k_maxcircle = []
    time_maxcircle = []
    k_mincircle = []
    time_mincircle = []
    
    k_x= []
    time_x = []
    k_maxcircle_1 = []
    time_maxcircle_1 = []
    k_mincircle_1 = []
    time_mincircle_1 = []
    
    k_y= []
    time_y = []
    k_maxcircle_2 = []
    time_maxcircle_2 = []
    k_mincircle_2 = []
    time_mincircle_2 = []
    
    for i in range(100):
        #np.random.seed(i)
        #data_raw = np.random.rand(1000, 2) * 100

        start = time()
        number_disk, center_locations, cluster_labels, centerindex_allocation = DR_1(data_raw, R, num_populations)
        end1 = time()
        t1 = end1 - start
        time_x.append(t1)
        k_x.append(number_disk)
        
        new_number_disk = Enhanced_DR(data_raw, R, number_disk, center_locations, cluster_labels, centerindex_allocation)

        end2 = time()
        t2 = end2 - start
        time_y.append(t2)
        k_y.append(new_number_disk)

    count_x = pd.value_counts(k_x)
    aver_time_x = sum(time_x)/len(time_x)
    print('count_random_1={}'.format(count_x))
    print('aver_time_random_1={}'.format(aver_time_x))
    
    count_y = pd.value_counts(k_y)
    aver_time_y = sum(time_y)/len(time_y)
    print('count_random_2={}'.format(count_y))
    print('aver_time_random_2={}'.format(aver_time_y))
