import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

def visualize_tax3d_data(data_dict):
    # Extract data from dictionary
    pc_action = data_dict['pc_action']
    pc_anchor = data_dict['pc_anchor']
    flow = data_dict['flow']

    # Create Open3D point clouds
    pcd_action = o3d.geometry.PointCloud()
    pcd_action.points = o3d.utility.Vector3dVector(pc_action)
    pcd_action.paint_uniform_color([1, 0, 0])  # Red color for action point cloud

    pcd_anchor = o3d.geometry.PointCloud()
    pcd_anchor.points = o3d.utility.Vector3dVector(pc_anchor)
    pcd_anchor.paint_uniform_color([0, 1, 0])  # Green color for anchor point cloud

    # Create Open3D point cloud for pc_action + flow
    pcd_action_flow = o3d.geometry.PointCloud()
    pcd_action_flow.points = o3d.utility.Vector3dVector(pc_action + flow)
    pcd_action_flow.paint_uniform_color([0, 0, 1])  # Blue color for action + flow point cloud

    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([pcd_action, pcd_anchor, pcd_action_flow, coordinate_frame])

# Example usage
if __name__ == "__main__":
    # Example data dictionary
    data_dict = {
        'pc_action': np.random.rand(100, 3),
        'pc_anchor': np.random.rand(100, 3),
        'seg_action': np.random.randint(0, 5, 100),
        'seg_anchor': np.random.randint(0, 5, 100),
        'flow': np.random.rand(100, 3) * 0.1
    }
    visualize_tax3d_data(data_dict)