import numpy as np
import trimesh

mesh_file = np.load("../small-dataset/17/final-12.npz")
pointcloud = mesh_file["pointcloud"]
mesh = trimesh.PointCloud(pointcloud)
trimesh.Scene(mesh).show()