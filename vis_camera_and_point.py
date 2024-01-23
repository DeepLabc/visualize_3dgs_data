import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plyfile
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parser_args():
    parser = argparse.ArgumentParser(description='visualize pose and point cloud from 3dgs')
    parser.add_argument('--camera_path', required=True, help='Path to the camera file(camera.json)')
    parser.add_argument('--point_path', required=True, help="Path to the point_cloud file(points3D.ply)")
    args = parser.parse_args()
    return args


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        origin = np.array([0, 0, 0])
        axis_length = 2.0
        self.ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, color='r', label='X-axis')
        self.ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, color='g', label='Y-axis')
        self.ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, color='b', label='Z-axis')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled,
                                1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                   vertex_transformed[4, :-1]]]
        meshes1 = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                   [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                   [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                   [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]]]

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=1))

        self.ax.add_collection3d(
            Poly3DCollection(meshes1, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.1))

    def add_points(self, points, downscale=1):
        point_cloud = points[::downscale]
        self.ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='black', marker='o')

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

    def save(self, output_file_path):
        plt.savefig(output_file_path)
        plt.close()


if __name__ == "__main__":
    args = parser_args()
    # init visualizer
    x_range = [-10, 10]
    y_range = [-10, 10]
    z_range = [-10, 10]
    visualizer = CameraPoseVisualizer(x_range, y_range, z_range)
    # process camera pose
    with open(args.camera_path, 'r') as json_file:
        data = json.load(json_file)
    for idx, info_dict in enumerate(data):
        position = np.array(info_dict["position"]).reshape(3, 1)
        rotation = np.array(info_dict["rotation"])

        pose = np.concatenate([rotation, position], axis=1)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])])

        visualizer.extrinsic2pyramid(pose, 'r', 1)
    # process point cloud
    ply_data = plyfile.PlyData.read(args.point_path)
    x = ply_data['vertex']['x']
    y = ply_data['vertex']['y']
    z = ply_data['vertex']['z']
    point_cloud = np.vstack((x, y, z)).T
    point_cloud = point_cloud[
        (point_cloud[:, 0] >= x_range[0]) & (point_cloud[:, 0] <= x_range[1]) &
        (point_cloud[:, 1] >= y_range[0]) & (point_cloud[:, 1] <= y_range[1]) &
        (point_cloud[:, 2] >= z_range[0]) & (point_cloud[:, 2] <= z_range[1])
        ]
    visualizer.add_points(point_cloud)

    visualizer.show()

