import numpy as np
import trimesh

mesh_file = np.load("../small-dataset-1/4/0653_U_Ori_target.npz")
pointcloud = mesh_file["pointcloud"]
mesh = trimesh.PointCloud(pointcloud)
trimesh.Scene(mesh).show()