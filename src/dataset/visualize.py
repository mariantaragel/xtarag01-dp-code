import numpy as np
import trimesh

mesh_file = np.load(
    "/home/marian/DP/removed-front-teeth-v3/point-clouds/0637_U_Ori_27_target.npz"
)
pointcloud = mesh_file["pointcloud"]
mesh = trimesh.PointCloud(pointcloud)
trimesh.Scene(mesh).show()
