"""
This script uses open3d v15.x to visualize a pointcloud stored in the .ply format
"""

import open3d as o3d
import sys

if len(sys.argv) < 2:
    print("enter filename of .ply file as argument (e.g. python vis.py pointcloud.ply)")
    exit(1)

filename = sys.argv[1]
pcd = o3d.io.read_point_cloud(filename)
o3d.visualization.draw_geometries([pcd])
