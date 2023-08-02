import os.path as osp
import pickle

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from data.utils.augmentation_3d import augment_and_scale_3d


class SemanticKITTIBase(Dataset):
    """SemanticKITTI dataset"""

    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    categories_gta = {
        'car': ['car', 'moving-car'],
        'high-vehicle': ['truck', 'moving-truck', 'bus', 'moving-bus'],
        'bike': ['bicycle', 'motorcycle'],
        'rider': ['bicyclist', 'motorcyclist', 'moving-bicyclist', 'moving-motorcyclist'],
        'person': ['person', 'moving-person'],
        'road': ['road', 'lane-marking'],
        'parking': ['parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'vegetation': ['vegetation'],
        'terrain': ['terrain'],
        'other-objects': ['fence', 'pole', 'traffic-sign', 'other-object'],
    }
    def __init__(self,
                 split,
                 preprocess_dir,
                 source,
                 merge_classes=False,
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir
        self.source = source
        print("Initialize SemanticKITTI dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            if  source == 'GTA':
                categories = self.categories_gta
            else:
                categories = None
            for cat_idx, cat_list in enumerate(categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticKITTISCN(SemanticKITTIBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 semantic_kitti_dir='',
                 source=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 bottom_crop=tuple(),  # 2D augmentation (also effects 3D)
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 use_aug=False,
                 ):
        super().__init__(split,
                         preprocess_dir,
                         source,
                         merge_classes=merge_classes,
                         )

        self.semantic_kitti_dir = semantic_kitti_dir

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl
        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.bottom_crop = bottom_crop
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.use_aug = use_aug
    def __getitem__(self, index):
        data_dict = self.data[index]

        points = data_dict['points'].copy()
        seg_label = data_dict['seg_labels'].astype(np.int64)
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        keep_idx = np.ones(len(points), dtype=np.bool)
        points_img = data_dict['points_img'].copy()
        img_path = osp.join(self.semantic_kitti_dir, data_dict['camera_path'])
        image = Image.open(img_path)

        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]

            # update image points
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            # update point cloud
            points = points[keep_idx]
            seg_label = seg_label[keep_idx]
        img_indices = points_img.astype(np.int64)
        out_dict['orig_img'] = np.moveaxis(np.array(image, dtype=np.float32, copy=False) / 255., -1, 0)
        out_dict['orig_img_indices'] = img_indices.copy()
        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['img_indices'] = img_indices

        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl)
        coords = coords.astype(np.int64)
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = np.ones([len(idxs[idxs]), 1], np.float32)  # simply use 1 as feature
        out_dict['seg_label'] = seg_label[idxs]
        out_dict['img_indices'] = out_dict['img_indices'][idxs]
        if self.use_aug:
            orig_coords = augment_and_scale_3d(points, self.scale, self.full_scale)
            # cast to integer
            orig_coords = orig_coords.astype(np.int64)
            # only use voxels inside receptive field
            orig_idxs = (orig_coords.min(1) >= 0) * (orig_coords.max(1) < self.full_scale)

            out_dict.update({
                'orig_coords': orig_coords[orig_idxs],
                'orig_feats': np.ones([len(orig_idxs), 1], np.float32),  # simply use 1 as feature
                'orig_points': points[orig_idxs],
                'orig_seg_label': seg_label[orig_idxs]
            })
        return out_dict

