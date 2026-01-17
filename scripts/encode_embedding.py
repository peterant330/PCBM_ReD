import torchvision
import os
import torch
from torch import nn, einsum
from open_clip import create_model_and_transforms
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from norm import IterNorm
import pickle
import open_clip
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm
import json
class CUB_data(Dataset):
    def __init__(self, root=None, transform = None, split = "train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        train = pd.read_pickle(os.path.join(self.root, 'class2images_train.p'))
        labelmapping = list(train.keys())
        self.labelmapping = labelmapping
        if split == "train":
            data = train
        elif split == "val":
            data = pd.read_pickle(os.path.join(self.root, 'class2images_val.p'))
        elif split == "test":
            data = pd.read_pickle(os.path.join(self.root, 'class2images_test.p'))
        self.data = []
        for i in range(len(labelmapping)):
            if labelmapping[i] in data:
                for j in data[labelmapping[i]]:
                    self.data.append((j, i))
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self.transform(Image.open(os.path.join(self.root, "images", sample[0])))
        label = sample[1]
        return image, label

def build_cbm(args):
    model, _, image_processor = create_model_and_transforms('ViT-L-14', pretrained='openai', device='cuda')
    if args.data == 'food':
        ds = CUB_data(root="datasets/food", split="train", transform=image_processor)
    elif args.data == 'cub':
        ds = CUB_data(root="datasets/cub", split="train", transform=image_processor)
    elif args.data == 'ham':
        ds = CUB_data(root="datasets/ham", split="train", transform=image_processor)
    elif args.data == 'dtd':
        ds = CUB_data(root="datasets/dtd", split="train", transform=image_processor)
    elif args.data == 'RESISC':
        ds = CUB_data(root="datasets/RESISC", split="train", transform=image_processor)
    elif args.data == 'cifar10':
        ds = CUB_data(root="datasets/cifar10", split="train", transform=image_processor)
    elif args.data == 'cifar100':
        ds = CUB_data(root="datasets/cifar100", split="train", transform=image_processor)
    elif args.data == 'ucf':
        ds = CUB_data(root="datasets/ucf", split="train", transform=image_processor)
    elif args.data == 'aircraft':
        ds = CUB_data(root="datasets/aircraft", split="train", transform=image_processor)
    elif args.data == 'flower':
        ds = CUB_data(root="datasets/flower", split="train", transform=image_processor)
    elif args.data == 'imagenet':
        ds = CUB_data(root="datasets/imagenet/", split="train", transform=image_processor)
    
    dl = DataLoader(ds, batch_size=64, num_workers=8, shuffle=False, drop_last=False)

    target_train = []
    image_matrix = torch.zeros([0, 768]).cuda()
    for batch in tqdm(dl):
        image, target = batch[0].cuda(), batch[1].cuda()
        with torch.inference_mode():
            image_features = model.encode_image(image)
            #image_features /= image_features.norm(dim=-1, keepdim=True)
            image_matrix = torch.cat([image_matrix, image_features], dim=0)
            target_train.extend(target.tolist())

    np.save(f"file/{args.data}/image_emb_train.npy", image_matrix.cpu().numpy())

    if args.data == 'food':
        ds = CUB_data(root="datasets/food", split="val", transform=image_processor)
    elif args.data == 'cub':
        ds = CUB_data(root="datasets/cub", split="val", transform=image_processor)
    elif args.data == 'ham':
        ds = CUB_data(root="datasets/ham", split="val", transform=image_processor)
    elif args.data == 'dtd':
        ds = CUB_data(root="datasets/dtd", split="val", transform=image_processor)
    elif args.data == 'RESISC':
        ds = CUB_data(root="datasets/RESISC", split="val", transform=image_processor)
    elif args.data == 'cifar10':
        ds = CUB_data(root="datasets/cifar10", split="val", transform=image_processor)
    elif args.data == 'cifar100':
        ds = CUB_data(root="datasets/cifar100", split="val", transform=image_processor)
    elif args.data == 'ucf':
        ds = CUB_data(root="datasets/ucf", split="val", transform=image_processor)
    elif args.data == 'aircraft':
        ds = CUB_data(root="datasets/aircraft", split="val", transform=image_processor)
    elif args.data == 'flower':
        ds = CUB_data(root="datasets/flower", split="val", transform=image_processor)
    elif args.data == 'imagenet':
        ds = CUB_data(root="datasets/imagenet", split="val", transform=image_processor)

    dl = DataLoader(ds, batch_size=8, num_workers=8, shuffle=False, drop_last=False)
    image_matrix = torch.zeros([0, 768]).cuda()
    target_val = []
    with torch.no_grad():
        for batch in tqdm(dl):
            image, target = batch[0].cuda(), batch[1].cuda()
            with torch.inference_mode():
                image_features = model.encode_image(image)
                #image_features /= image_features.norm(dim=-1, keepdim=True)
                image_matrix = torch.cat([image_matrix, image_features], dim=0)
                target_val.extend(target.tolist())

    np.save(f"file/{args.data}/image_emb_val.npy", image_matrix.cpu().numpy())

    
    if args.data == 'food':
        ds = CUB_data(root="datasets/food", split="test", transform=image_processor)
    elif args.data == 'cub':
        ds = CUB_data(root="datasets/cub", split="test", transform=image_processor)
    elif args.data == 'ham':
        ds = CUB_data(root="datasets/ham/", split="test", transform=image_processor)
    elif args.data == 'dtd':
        ds = CUB_data(root="datasets/dtd", split="test", transform=image_processor)
    elif args.data == 'RESISC':
        ds = CUB_data(root="datasets/RESISC", split="test", transform=image_processor)
    elif args.data == 'cifar10':
        ds = CUB_data(root="datasets/cifar10", split="test", transform=image_processor)
    elif args.data == 'cifar100':
        ds = CUB_data(root="datasets/cifar100", split="test", transform=image_processor)
    elif args.data == 'ucf':
        ds = CUB_data(root="datasets/ucf101", split="test", transform=image_processor)
    elif args.data == 'aircraft':
        ds = CUB_data(root="datasets/aircraft", split="test", transform=image_processor)
    elif args.data == 'flower':
        ds = CUB_data(root="datasets/flower", split="test", transform=image_processor)
    elif args.data == 'imagenet':
        ds = CUB_data(root="datasets/imagenet/", split="test", transform=image_processor)

    dl = DataLoader(ds, batch_size=64, num_workers=8, shuffle=False, drop_last=False)
    image_matrix = torch.zeros([0, 768]).cuda()
    target_test = []
    for i, batch in enumerate(dl):
        image, target = batch[0].cuda(), batch[1].cuda()
        with torch.inference_mode():
            image_features = model.encode_image(image)
            #image_features /= image_features.norm(dim=-1, keepdim=True)
            image_matrix = torch.cat([image_matrix, image_features], dim=0)
            target_test.extend(target.tolist())

    np.save(f"file/{args.data}/image_emb_test.npy", image_matrix.cpu().numpy())

    target_test = target_val
    with open(f"file/{args.data}/target.pkl", "wb") as f:
        pickle.dump((target_train, target_val, target_test), f)



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="food")
    args = parser.parse_args()
    build_cbm(args)

