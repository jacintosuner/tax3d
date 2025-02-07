import os
import sys
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import numpy as np
from pyk4a import PyK4APlayback
from pyk4a.calibration import CalibrationType
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import scripts.vis_tax3d_data as vis
sys.path.append(os.path.abspath("../third_party/Grounded-SAM-2"))
from gsam_wrapper import GSAM2


def random_se3() -> np.ndarray:
    # Generate a random rotation matrix
    random_rotation = R.random().as_matrix()
    
    # Generate a random translation vector
    random_translation = np.random.uniform(-1, 1, size=3)
    
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = random_rotation
    transformation_matrix[:3, 3] = random_translation
    
    return transformation_matrix


class VideotoTax3D:
    """
        This class processes a directory of mkv videos into TAX3D training data.
        The input_data_dir should either be an mkv video or a folder with mkv videos (captured using PyK4A if a Kinect camera is used).
        Point clouds are converted to the world frame to facilitate data augmentation, which rotates the action object along the z-axis (vertical axis in the world frame).
        Future versions aim to support additional raw input formats for greater robustness.
    """
    def __init__(self, cfg: OmegaConf):
        self.input_data_dir = cfg.preprocessing.input_data_dir
        self.output_data_dir = cfg.preprocessing.output_data_dir
        self.camera2world_file = cfg.preprocessing.camera2world_file
        self.technique = cfg.preprocessing.technique
        self.object_name = cfg.preprocessing.object_name
        self.num_objects = cfg.preprocessing.get('num_objects', 1)  # Add this line
        self.gsam2 = GSAM2(debug=self.debug)
        self.to_world_frame = cfg.preprocessing.world_frame
        self.debug = cfg.preprocessing.debug

        # Create output_data_dir if it doesn't exist
        os.makedirs(self.output_data_dir, exist_ok=True)

    def _load_camera2world(self) -> np.ndarray:
        """
        This function loads the camera-to-world transformation matrix.
        Returns:
            torch.Tensor: The camera-to-world transformation matrix.
        """
        if self.camera2world_file is None:
            print("No camera2world file provided. Using identity matrix.")
            return np.eye(4)
        return np.load(self.camera2world_file)
    
    def _get_raw_data_from_mkv(self, input_file: str) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        playback = PyK4APlayback(input_file)
        playback.open()

        frames = []
        depths = []
        K = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
        while True:
            try:
                capture = playback.get_next_capture()
                frames.append(capture.color[:, :, :3][:, :, ::-1])
                depths.append(capture.transformed_depth)
            except EOFError:
                break
        playback.close()

        return frames, depths, K
    
    def _frame_to_pc(self, frame: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        This function converts a frame and depth map to a point cloud.
        Args:
            frame (np.ndarray): The RGB frame.
            depth (np.ndarray): The depth map.
            K (np.ndarray): The camera intrinsic matrix.
        Returns:
            np.ndarray: The point cloud. Shape: (N, 3)
        """
        h, w = frame.shape[:2]
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        xx = (xx - K[0, 2]) * zz / K[0, 0]
        yy = (yy - K[1, 2]) * zz / K[1, 1]
        pc = np.stack([xx, yy, zz], axis=-1)

        # Point cloud is converted to world frame so that the action object can be rotated along the z-axis (and still be in the table plane)
        if self.to_world_frame:
            Tcamera2world = self._load_camera2world()
            pc = (Tcamera2world[:3, :3] @ pc.T + Tcamera2world[:3, 3:]).T
        return pc
    
    def _process_using_final_frame_and_se3_transform(self, frames: list[np.ndarray], depths: list[np.ndarray], K: np.ndarray) -> dict:
        """
        This function processes the raw data using the final frame of the video and
        applies an SE3 transform to the final action point cloud as a way to get an initial action point cloud.

        Args:
            frames (list): A list of video frames.
            depths (list): A list of depth maps corresponding to the frames.
            K (numpy.ndarray): The camera intrinsic matrix.
        Returns:
            dict: A dictionary containing the processed data with the following keys:
            - 'pc_action' (np.ndarray): Action points in their starting position. Shape: (N_action, 3)
            - 'pc_anchor' (np.ndarray): Anchor points in their starting position. Shape: (N_anchor, 3)
            - 'seg_action' (np.ndarray): Segmentation mask for the action points. Shape: (N,)
            - 'seg_anchor' (np.ndarray): Segmentation mask for the anchor points. Shape: (N,)
            - 'flow' (np.ndarray): Flow field from initial action to final action point clouds. Shape: (N_action, 3)
        """
        
        # GSAM inference
        gsam2 = GSAM2(debug=self.debug)
        masks, scores, logits, confidences, labels, input_boxes = gsam2.get_masks_image(self.object_name, frames[-1])
        if masks is None:
            return None
        segmentation, _ = gsam2.filter_masks(masks, labels, self.num_objects)
        segmentation.flatten()
        segmentation = segmentation.squeeze().reshape(-1)

        # Get the point cloud
        final_pc = self._frame_to_pc(frames[-1], depths[-1], K)

        # Apply the SE3 transform
        T = random_se3()
        initial_pc = (T[:3, :3] @ final_pc.T + T[:3, 3:]).T
        
        # Save the processed data
        return {
            'pc_action': initial_pc[segmentation == 1],
            'pc_anchor': final_pc[segmentation == 0], # we consider the anchor pc does not move in this case, so we can take the final pc as the anchor pc
            'seg_action': segmentation,
            'seg_anchor': 1 - segmentation,
            'flow': final_pc[segmentation == 1] - initial_pc[segmentation == 1],
        }
    

    def process(self) -> None:
        """
            This function processes the raw data into a dataset that can be used for training.
        """
        
        mkv_files = list(Path(self.input_data_dir).glob('*.mkv'))
        for i, mkv_file in enumerate(mkv_files):
            print("----------------------------- Processing file:", mkv_file)

            # Get the frames, depths and K from the mkv_file
            frames, depths, K = self._get_raw_data_from_mkv(mkv_file)

            # Process the data using the specified technique
            if self.technique == "final_frame_and_se3_transform":
                item = self._process_using_final_frame_and_se3_transform(frames, depths, K)
            if item is None:
                continue

            # Save the processed data
            output_file = Path(self.output_data_dir) / f"demo_{i}.npz"
            if self.debug:
                vis.visualize_tax3d_data(item)
            np.savez(output_file, **item)


@hydra.main(config_path="../configs/dataset", config_name="real_world", version_base="1.3")
def main(cfg: OmegaConf) -> None:
    # Convert cfg to dict if needed to make mutable
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_dict)
    
    # Debug print for Tax3D config
    if cfg.preprocessing.debug:
        print("Tax3D config:", OmegaConf.to_yaml(cfg))
    
    real_data_preprocessor = VideotoTax3D(cfg)
    real_data_preprocessor.process()

if __name__ == "__main__":
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    main()
