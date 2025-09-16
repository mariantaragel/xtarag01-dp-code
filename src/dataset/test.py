import numpy as np
import trimesh

mesh_file = np.load(
    "/home/marian/DP/data-remove/small-dataset-remove/4/0511_U_Ori_target.npz"
)
pointcloud = mesh_file["pointcloud"]
mesh = trimesh.PointCloud(pointcloud)
trimesh.Scene(mesh).show()
