import numpy as np
import open3d as o3d
import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from datasets.point_clouds_utils import PointCloudLoader

class ALITAPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud propertiers, such as ground_plane_level. Must be defined in inherited classes.
        self.ground_plane_level = -1.6

    def read_pc(self, file_pathname):
        pcd = o3d.io.read_point_cloud(file_pathname)
        xyz = np.asarray(pcd.points)
        return xyz