import os.path as osp

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path


class GTABase(Dataset):
    id_to_class_name = {
        -1: "license plate",
        0: "unlabeled",
        1: "ego-vehicle",
        2: "rectification-border",
        3: "out-of-roi",
        4: "static",
        5: "dynamic",
        6: "ground",
        7: "road",
        8: "sidewalk",
        9: "parking",
        10: "rail-track",
        11: "building",
        12: "wall",
        13: "fence",
        14: "ground-rail",
        15: "bridge",
        16: "tunnel",
        17: "pole",
        18: "pole-group",
        19: "traffic-light",
        20: "traffic-sign",
        21: "vegetation",
        22: "terrain",
        23: "sky",
        24: "person",
        25: "rider",
        26: "car",
        27: "truck",
        28: "bus",
        29: "caravan",
        30: "trailer",
        31: "train",
        32: "motorcycle",
        33: "bicycle",
    }
    class_name_to_id = {v: k for k, v in id_to_class_name.items()}
    categories_skitti = {
        'car': ['car'],
        'high-vehicle': ['truck', 'bus'],
        'bike': ['bicycle', 'motorcycle'],
        'rider': ['rider'],
        'person': ['person'],
        'road': ['road'],
        'parking': ['parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'vegetation': ['vegetation'],
        'terrain': ['terrain'],
        'other-objects': ['fence', 'pole', 'traffic-sign', 'traffic-light'],
    }
    def __init__(self, root, list_path, target, split, merge_classes):
        self.root = Path(root)
        self.split = split
        print("Initialize GTA dataloader")
        assert isinstance(split, tuple)
        print('Load', split)
        self.list_path = list_path.format(self.split[0])
        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            if target == 'SemanticKITTI':
                categories = self.categories_skitti
            else:
                categories = None
            for cat_idx, cat_list in enumerate(categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(categories.keys())
        else:
            self.label_mapping = None
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        self.data = []
        for name in self.img_ids:
            img, label = self.get_metadata(name)
            self.data.append((img, label, name))
    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class GTASCN(GTABase):
    def __init__(self,
                 split,
                 root,
                 list_path,
                 target,
                 merge_classes,
                 resize=(480, 302),
                 image_normalizer=None,
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 ):
        super().__init__(root,
                         list_path,
                         target,
                         split,
                         merge_classes
                         )

        # point cloud parameters
        self.split = split
        # image parameters
        self.resize = resize
        self.image_normalizer = image_normalizer

        # data augmentation
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file
    def __getitem__(self, index):
        data_dict = self.data[index]

        out_dict = {}
        img_path = osp.join(data_dict[0])
        image = Image.open(img_path)
        label_path = osp.join(data_dict[1])
        label = Image.open(label_path)
        if self.resize:
            if not image.size == self.resize:
                # check if we do not enlarge downsized images
                assert image.size[0] > self.resize[0]
                resize = (round(image.size[0]/image.size[1]*self.resize[1]),self.resize[1])
                # resize image
                image = image.resize(resize, Image.BILINEAR)
                label = label.resize(resize, Image.NEAREST)
                left = int(np.random.rand() * (image.size[0] + 1 - self.resize[0]))
                right = left + self.resize[0]
                top = image.size[1] - self.resize[1]
                bottom = image.size[1]
                image = image.crop((left, top, right, bottom))
                label = label.crop((left, top, right, bottom))
        out_dict['orig_img'] = np.moveaxis(np.array(image, dtype=np.float32, copy=False) / 255., -1, 0)
        out_dict['orig_label_dense'] = self.label_mapping[np.array(label)]
        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        label = np.array(label)
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            label = np.ascontiguousarray(np.fliplr(label))
        label_dense = self.label_mapping[label]
        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['label_dense'] = label_dense
        return out_dict


