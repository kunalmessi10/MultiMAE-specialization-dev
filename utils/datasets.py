# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from utils import create_transform

from .data_constants import (IMAGE_TASKS, IMAGENET_DEFAULT_MEAN,
                             IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN,
                             IMAGENET_INCEPTION_STD)
from .dataset_folder import ImageFolder, MultiTaskImageFolder


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        trans = [transforms.RandomResizedCrop(args.input_size)]
        if args.hflip > 0.0:
            trans.append(transforms.RandomHorizontalFlip(args.hflip))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))])

        self.transform = transforms.Compose(trans)

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class DataAugmentationForMultiMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        self.hflip = args.hflip

    def __call__(self, task_dict):
        flip = random.random() < self.hflip # Stores whether to flip all images or not
        ijhw = None # Stores crop coordinates used for all tasks
        
        # Crop and flip all tasks randomly, but consistently for all tasks
        for task in task_dict:
            if task not in IMAGE_TASKS:
                continue
            if ijhw is None:
                # Official MAE code uses (0.2, 1.0) for scale and (0.75, 1.3333) for ratio
                ijhw = transforms.RandomResizedCrop.get_params(
                    task_dict[task], scale=(0.2, 1.0), ratio=(0.75, 1.3333)
                )
            i, j, h, w = ijhw
            task_dict[task] = TF.crop(task_dict[task], i, j, h, w)
            task_dict[task] = task_dict[task].resize((self.input_size, self.input_size))
            if flip:
                task_dict[task] = TF.hflip(task_dict[task])
                
        # Convert to Tensor
        for task in task_dict:
            if task in ['depth']:
                img = torch.Tensor(np.array(task_dict[task]) / 2 ** 16)
                img = img.unsqueeze(0)  # 1 x H x W
            elif task in ['rgb']:
                img = TF.to_tensor(task_dict[task])
                img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
            elif task in ['semseg', 'semseg_coco']:
                # TODO: add this to a config instead
                # Rescale to 0.25x size (stride 4)
                scale_factor = 0.25
                img = task_dict[task].resize((int(self.input_size * scale_factor), int(self.input_size * scale_factor)))
                # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
                img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)
                
            task_dict[task] = img
        
        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        #repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

class MixtureMultiTaskDataset(torch.utils.data.Dataset):
    """Map-style weighted mixture over multiple pretraining datasets."""

    def __init__(
            self,
            datasets: Sequence[torch.utils.data.Dataset],
            weights: Sequence[float],
            replacement: Optional[Sequence[bool]] = None,
            samples_per_epoch: Optional[int] = None,
            seed: int = 0,
    ):
        if weights is None:
            raise ValueError("mixture_weights must be provided when mixture_data_paths is set")
        if len(datasets) == 0:
            raise ValueError("mixture_data_paths must contain at least one dataset path")
        if len(datasets) != len(weights):
            raise ValueError("mixture_data_paths and mixture_weights must have the same length")
        if any(weight <= 0 for weight in weights):
            raise ValueError("mixture_weights must all be positive")
        if replacement is None:
            replacement = [True] * len(datasets)
        if len(datasets) != len(replacement):
            raise ValueError("mixture_data_paths and mixture_replacement must have the same length")

        self.datasets = list(datasets)
        self.weights = np.array(weights, dtype=np.float64)
        self.sampling_probs = self.weights / self.weights.sum()
        self.replacement = list(replacement)
        self.samples_per_epoch = samples_per_epoch or sum(len(dataset) for dataset in self.datasets)
        if self.samples_per_epoch <= 0:
            raise ValueError("mixture_samples_per_epoch must be positive")
        self.seed = seed
        self.epoch = 0
        self._plan_epoch = None
        self._source_indices = None
        self._sample_indices = None

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._plan_epoch = None

    def __len__(self):
        return self.samples_per_epoch

    @staticmethod
    def parse_replacement_flags(flags: Optional[Sequence[str]], num_datasets: int) -> Optional[List[bool]]:
        if flags is None:
            return None
        if len(flags) != num_datasets:
            raise ValueError("mixture_data_paths and mixture_replacement must have the same length")

        parsed = []
        for flag in flags:
            if isinstance(flag, bool):
                parsed.append(flag)
                continue
            normalized = flag.lower()
            if normalized in ('true', '1', 'yes', 'y'):
                parsed.append(True)
            elif normalized in ('false', '0', 'no', 'n'):
                parsed.append(False)
            else:
                raise ValueError("mixture_replacement values must be true/false")
        return parsed

    def _build_epoch_plan(self):
        if self._plan_epoch == self.epoch:
            return

        rng = np.random.default_rng(self.seed + self.epoch)
        self._source_indices = rng.choice(
            len(self.datasets), size=self.samples_per_epoch, p=self.sampling_probs
        ).astype(np.int64)
        self._sample_indices = np.empty(self.samples_per_epoch, dtype=np.int64)

        for dataset_idx, dataset in enumerate(self.datasets):
            positions = np.flatnonzero(self._source_indices == dataset_idx)
            if len(positions) == 0:
                continue

            if self.replacement[dataset_idx]:
                self._sample_indices[positions] = rng.integers(0, len(dataset), size=len(positions))
                continue

            source_indices = []
            while len(source_indices) < len(positions):
                source_indices.extend(rng.permutation(len(dataset)).tolist())
            self._sample_indices[positions] = np.array(source_indices[:len(positions)], dtype=np.int64)

        self._plan_epoch = self.epoch

    def resolve_source_index(self, index: int) -> Tuple[int, int]:
        index = int(index)
        self._build_epoch_plan()
        return int(self._source_indices[index]), int(self._sample_indices[index])

    def __getitem__(self, index: int):
        dataset_idx, sample_idx = self.resolve_source_index(index)
        return self.datasets[dataset_idx][sample_idx]


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)

def build_multimae_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    mixture_data_paths = getattr(args, 'mixture_data_paths', None)
    if mixture_data_paths is not None:
        datasets = [
            MultiTaskImageFolder(data_path, args.all_domains, transform=transform)
            for data_path in mixture_data_paths
        ]
        dataset = MixtureMultiTaskDataset(
            datasets=datasets,
            weights=getattr(args, 'mixture_weights', None),
            replacement=MixtureMultiTaskDataset.parse_replacement_flags(
                getattr(args, 'mixture_replacement', None), len(datasets)
            ),
            samples_per_epoch=getattr(args, 'mixture_samples_per_epoch', None),
            seed=args.seed,
        )
        source_lengths = ', '.join(str(len(source_dataset)) for source_dataset in datasets)
        print(f"Using mixture dataset with source lengths [{source_lengths}], "
              f"weights {args.mixture_weights}, replacement {dataset.replacement}, "
              f"samples_per_epoch {len(dataset)}")
        return dataset
    return MultiTaskImageFolder(args.data_path, args.all_domains, transform=transform)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
