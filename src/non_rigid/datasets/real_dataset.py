import os
from pathlib import Path

import numpy as np
import torch
import cv2
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as T
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import torch_geometric.transforms as tgt
from pytorch3d.transforms import Transform3d, Translate

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import scripts.vis_tax3d_data as vis

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd
from non_rigid.utils.augmentation_utils import ball_occlusion, plane_occlusion, maybe_apply_augmentations


class RealDataset(data.Dataset):
    """
        This Dataset class is used during the training process to augment basic tax3d training data.
    """
    def __init__(self, root, dataset_cfg, split):
        super().__init__()
        self.root = root
        self.split = split
        self.dataset_dir = self.root / self.split
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        print(self.num_demos)
        self.dataset_cfg = dataset_cfg

        # determining dataset size - if not specified, use all demos in directory once
        size = self.dataset_cfg.train_size if "train" in self.split else self.dataset_cfg.val_size
        if size is not None:
            self.size = size
        else:
            self.size = self.num_demos

        # setting sample sizes
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        self.world_frame = self.dataset_cfg.world_frame

    def __len__(self):
        return self.size
    
    def __getitem__(self, index: int) -> dict:

        # loop over the dataset multiple times, allowing for arbitrary dataset and batch sizes using data augmentation e.g.
        filter_index = index % self.num_demos

        # load data
        demo = np.load(self.dataset_dir / f"demo_{filter_index}.npz", allow_pickle=True)

        pc_action = torch.as_tensor(demo["pc_action"]).float()
        seg_action = torch.as_tensor(demo["seg_action"]).int()
        pc_anchor = torch.as_tensor(demo["pc_anchor"]).float()
        seg_anchor = torch.as_tensor(demo["seg_anchor"]).int()
        flow = torch.as_tensor(demo["flow"]).float()

        speed_factor = torch.ones(1)
        rot = torch.zeros(3)
        trans = torch.zeros(3)

        # downsampling action point cloud
        if self.sample_size_action > 0 and pc_action.shape[0] > self.sample_size_action:
            pc_action, pc_action_indices = downsample_pcd(pc_action.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc = pc_action.squeeze(0)
            seg_action = seg_action[pc_action_indices.squeeze(0)]
            flow = flow[pc_action_indices.squeeze(0)]

        # downsampling anchor point cloud
        pc_anchor, pc_anchor_indices = downsample_pcd(pc_anchor.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        pc_anchor = pc_anchor.squeeze(0)
        seg_anchor = seg_anchor[pc_anchor_indices.squeeze(0)]

        # apply some data augmentation

        points_action = pc_action + flow
        points_anchor = pc_anchor

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        goal_points_action = points_action - center
        goal_points_anchor = points_anchor - center

        # Transform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )

        goal_points_action = T1.transform_points(goal_points_action)
        goal_points_anchor = T1.transform_points(goal_points_anchor)
        T_goal2world = T1.inverse().compose(
            Translate(center.unsqueeze(0))
        )

        if self.dataset_cfg.action_context_center_type == "center":
            action_center = pc_action.mean(axis=0)
        elif self.dataset_cfg.action_context_center_type == "random":
            action_center = pc_action[np.random.choice(len(pc_action))]
        elif self.dataset_cfg.action_context_center_type == "none":
            action_center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")

        points_action = pc_action - action_center
        points_action = T0.transform_points(points_action)
        T_action2world = T0.inverse().compose(
            Translate(action_center.unsqueeze(0))
        )
        # Get the flow
        gt_flow = goal_points_action - points_action

        item = {
        'pc': goal_points_action, # action points in goal position, used for training a model that predicts the action point cloud
        'pc_action': points_action, # action points in starting position for context
        'pc_anchor': goal_points_anchor, # anchor points in goal position
        'seg': seg_action, # segmentation mask for the action points
        'seg_anchor': seg_anchor, # segmentation mask for the anchor points
        'flow': gt_flow, # flow field from initial action to final action point clouds
        'T_goal2world': T_goal2world.get_matrix().squeeze(0),
        'T_action2world': T_action2world.get_matrix().squeeze(0)
        # 'speed_factor': speed_factor, # used for simulation purposes
        # 'rot': rot, # used for simulation purposes
        # 'trans': trans, # used for simulation purposes
        }
        return item
    

class RealDataModule(L.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, num_workers, dataset_cfg):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg
        
        data_dir = os.path.expanduser(dataset_cfg.data_dir)
        self.root = data_dir

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        self.stage = stage

        if self.stage == "fit" or self.stage is None:
            self.train_dataset = RealDataset(self.root, self.dataset_cfg, "train")
            self.val_dataset = RealDataset(self.root, self.dataset_cfg, "val")
        
        # TODO fit, world frame,... (check proc_cloth_flow.py)
        # TODO: OOD dataset? (ask Eric)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "fit" else False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        return val_dataloader, val_ood_dataloader
    

# custom collate function to handle deform params
def cloth_collate_fn(batch):
    # batch is a list of dictionaries
    # we need to convert it to a dictionary of lists
    keys = batch[0].keys()
    out = {k: None for k in keys}
    for k in keys:
        if k == "deform_params":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out