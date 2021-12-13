# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import csv
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, pil_loader, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class SACustomDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2021,
        transform=None,
        target_transform=None,
        category='name',
        loader= pil_loader,
        extra_args = None
    ):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        self.nb_classes = None
        self.train = train
        self.samples = []
        model_dir = extra_args.finetune
        if train:
            path_to_manifest  = extra_args.manifest_dirs['train_manifest']
            self.images_dir = extra_args.image_dirs['train_images_dir']
        else:
            path_to_manifest  = extra_args.manifest_dirs['test_manifest']
            self.images_dir = extra_args.image_dirs['test_images_dir']

        categories = {}
        path_to_classes = 'model/classes_mapper.json'
        with open(path_to_classes, 'r') as fp:
            classes = json.load(fp)
            self.nb_classes = len(classes)

        with open(path_to_manifest, 'r') as fp:
            reader = csv.reader(fp, delimiter=',')
            print(path_to_manifest)

            self.samples = [
                (self.make_img_path(row[0]), int(row[1])) for row in reader
            ]

    def make_img_path(self, img_name):
        return os.path.join(self.images_dir, img_name)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'SACustomDataset':
        dataset = SACustomDataset(
            root = args.data_path,
            train=is_train,
            year=2021,
            category='name',
            transform=transform,
            extra_args = args
        )
        nb_classes = dataset.nb_classes
    print(dataset)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
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
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4
            )
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3
                             ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
