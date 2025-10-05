import numpy as np
import trimesh

mesh_file = np.load(
    "./0254_U_Ori_target.npz"
    # "/home/marian/DP/removed-front-teeth-v2/point-clouds/49/0254_U_Ori_target.npz"
)
pointcloud = mesh_file["pointcloud"]
print(pointcloud)
# mesh = trimesh.PointCloud(pointcloud)
# trimesh.Scene(mesh).show()
