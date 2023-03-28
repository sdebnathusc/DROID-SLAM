import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d
import os

from lietorch import SE3

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.005

    all_pts, all_clr = [], []

    with torch.no_grad():

        with video.get_lock():
            t = video.counter.value 
            dirty_index, = torch.where(video.dirty.clone())
            dirty_index = dirty_index

        if len(dirty_index) == 0:
            return

        video.dirty[dirty_index] = False

        # convert poses to 4x4 matrix
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)
        Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
        
        count = droid_backends.depth_filter(
            video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))
        
        for i in range(len(dirty_index)):
            pose = Ps[i]
            ix = dirty_index[i].item()

            if ix in droid_visualization.cameras:
                del droid_visualization.cameras[ix]

            if ix in droid_visualization.points:
                del droid_visualization.points[ix]

            ### add camera actor ###
            cam_actor = create_camera_actor(True)
            cam_actor.transform(pose)
            droid_visualization.cameras[ix] = cam_actor

            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
            
            ## add point actor ###
            point_actor = create_point_actor(pts, clr)
            droid_visualization.points[ix] = point_actor

            # append
            all_pts.extend(pts.tolist())
            all_clr.extend(clr.tolist())

        droid_visualization.ix += 1
    all_pts_np = np.array(all_pts)
    all_clr_np = np.array(all_clr)
    # print("Final shape: ", all_pts_np.shape, " : ", all_clr_np.shape)
    all_point_actor = create_point_actor(all_pts_np, all_clr_np)
    return all_point_actor
