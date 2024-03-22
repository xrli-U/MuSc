import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import random

_CLASSNAMES = ["01", "02", "03"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class BTADDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        clip_transformer=None,
        k_shot=0,
        random_seed=42,
        divide_num=1,
        divide_iter=0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        if divide_num > 1:
            # divide into subsets
            self.data_to_iterate = self.sub_datasets(self.data_to_iterate, divide_num, divide_iter, random_seed)

        if k_shot > 0:
            # few-shot
            torch.manual_seed(random_seed)
            if k_shot >= len(self.data_to_iterate):
                pass
            else:
                indices = torch.randint(0, len(self.data_to_iterate), (k_shot,))
                self.data_to_iterate = [self.data_to_iterate[i] for i in indices]
        if clip_transformer is None:
            self.transform_img = [
                transforms.Resize((resize,resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img = transforms.Compose(self.transform_img)
        else:
            self.transform_img = clip_transformer
            
        self.transform_mask = [
            transforms.Resize((resize,resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)
    
    def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
        # uniform division
        if divide_num == 0:
            return full_datasets
        random.seed(random_seed)

        id_dict = {}
        for i in range(len(full_datasets)):
            anomaly_type = full_datasets[i][2].split('/')[-2]
            if anomaly_type not in id_dict.keys():
                id_dict[anomaly_type] = []
            id_dict[anomaly_type].append(i)

        sub_id_list = []
        for k in id_dict.keys():
            type_id_list = id_dict[k]
            random.shuffle(type_id_list)
            devide_list = [type_id_list[i:i+divide_num] for i in range(0, len(type_id_list), divide_num)]
            sub_list = [devide_list[i][divide_iter] for i in range(len(devide_list)) if len(devide_list[i])>divide_iter]
            sub_id_list.extend(sub_list)

        return [full_datasets[id] for id in sub_id_list]

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])
    
        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly != "ok"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        
        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.split == DatasetSplit.TEST and anomaly != "ok":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["ok"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "ok":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
