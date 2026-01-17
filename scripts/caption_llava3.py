import torchvision
import argparse
import pandas as pd
from torch.utils.data import Dataset
import pickle
import os
import sys
sys.path.append("./models")
from tqdm import tqdm
from huggingface_hub import login

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

login(token = "")
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
class caltech_data(Dataset):
    def __init__(self, root=None, transform = None, split = "train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        data = pd.read_csv(os.path.join(self.root, 'split.txt'), delimiter = ',', header=None)
        if split == "train":
            self.data = data[data.iloc[:,2] == 1]
        else:
            self.data = data[data.iloc[:,2] == 0]
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image = self.transform(Image.open(os.path.join(self.root, sample[0])))
        label = sample[1]
        return image, label
import numpy as np
def get_idx(ds):
    K = 10
    n = 1000
    train_y = np.array(ds.targets)
    idx = np.arange(10000)
    idx = idx.reshape((n, -1))
    np.random.seed(0)
    ignore = [np.random.shuffle(x) for x in idx]

    idx = idx[:, :K].flatten()

    corr = np.concatenate([[i] * K for i in range(n)])
    y = train_y[idx]

    # CORRECT DISTRIBUTION IF NOT UNIFORM
    foo = (y == corr)
    i = -1
    for val in tqdm(foo):
        i += 1
        if not val:
            while train_y[idx[i]] < corr[i]:
                idx[i] += 1
            while train_y[idx[i]] > corr[i]:
                idx[i] -= 1

    y = train_y[idx]
    classes = np.arange(1000)
    classes_idx = np.argwhere(np.in1d(y, classes)).flatten()
    idx = idx[classes_idx]
    return idx
def rank_caption(args):
    with open(os.path.join(args.save, args.data, "rank.pkl"), "rb") as f:
        rank = pickle.load(f)
    idx_set = []
    for i in range(len(rank.keys())):
        idx_set.extend(rank[i][:20])
    idx_set = set(idx_set)

    if args.data == 'cub':
        ds = CUB_data(root="datasets/cub", split="train")
    elif args.data == 'food':
        ds = CUB_data(root="datasets/food", split="train")
    elif args.data == 'ham':
        ds = CUB_data(root="datasets/ham/", split="train")
    elif args.data == 'dtd':
        ds = CUB_data(root="datasets/dtd", split="train")
    elif args.data == 'RESISC':
        ds = CUB_data(root="datasets/RESISC", split="train")
    elif args.data == 'cifar10':
        ds = CUB_data(root="datasets/cifar10", split="train")
    elif args.data == 'cifar100':
        ds = CUB_data(root="datasets/cifar100", split="train")
    elif args.data == 'imagenet':
        ds = torchvision.datasets.ImageNet(root="datasets/imagenet/", split="train")
        #idx = get_idx(ds)
        #ds = torch.utils.data.Subset(ds, idx)
    elif args.data == 'ucf':
        ds = CUB_data(root="datasets/ucf", split="train")
    elif args.data == 'aircraft':
        ds = CUB_data(root="datasets/aircraft", split="train")
    elif args.data == 'flower':
        ds = CUB_data(root="datasets/flower", split="train")

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.tie_weights()
    if args.data == 'cub':
        prompt_text_1 = "What is the species of the bird? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Concisely summarize the appearance of the bird using one paragraph. Do not include the specific species name of the bird."
    elif args.data == 'food':
        prompt_text_1 = "What is the food? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Briefly summarize the appearance of the food in one paragraph. Do not mention what the object is."
    elif args.data == 'ham':
        prompt_text_1 = "What is the type of the pigmented lesion in the dermatoscopic image (e.g., Actinic keratoses, basal cell carcinoma, benign keratosis-like lesions, dermatofibroma, melanoma, melanocytic nevi, and vascular lesions)? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Briefly summarize the appearance of the pigmented lesion in one paragraph, do not include the specific type name."
    elif args.data in ['cifar10', 'cifar100', "imagenet"]:
        prompt_text_1 = "Describe the appearance of the object within the image. Only include the visual features that can help identify its category.  Use Chain-of-thought to reason."
        prompt_text_2 = "Briefly summarize the appearance of the object in one paragraph. Do not mention what the object is."
    elif args.data == 'dtd':
        prompt_text_1 = "What is the texture (e.g., blotchy, frilly)? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Briefly summarize the feature of the texture in one paragraph, do not mention the name of the texture."
    elif args.data == "RESISC":
        prompt_text_1 = "What is the scene class in the satellite image? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Briefly summarize the appearance of the scene in one paragraph. Do not mention what the scene is."
    elif args.data == "ucf":
        prompt_text_1 = "What is the action the characters are performing? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Briefly summarize visual pattern of the action using one paragraph. Do not mention what the action is."
    elif args.data == 'flower':
        prompt_text_1 = "What is the species of the flower? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Concisely summarize the appearance of the flower using one paragraph. Do not include the specific species name of the flower."
    elif args.data == 'aircraft':
        prompt_text_1 = "What is the model of the flight? Use chain of thought to reason and list all the visual features that lead to the conclusion."
        prompt_text_2 = "Concisely summarize the appearance of the flight using one paragraph. Do not include the specific model name."

    ################################################
    # preparation for the generation
    # unlikely that you need to change anything here
    with open(f"file/{args.data}/caption_llava.pkl", "rb") as f:
        caption_dict = pickle.load(f)
    for i in tqdm(list(idx_set)):
        if i in caption_dict:
            continue
        if args.data != "imagenet":
            image_path_or_url = os.path.join(ds.root, "images", ds.data[i][0])
        else:
            image_path_or_url = ds.imgs[i][0]
        image = Image.open(image_path_or_url)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": prompt_text_1}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1000)
        des = processor.decode(output[0])
        generated_text = des.split("assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip("\n").replace("\n\n",
                                                                                                               " ")

        conversation_history = [
            {"role": "user", "content": [
                {"type": "text",
                 "text": prompt_text_2 + "\n\n" + generated_text}
            ]},
        ]
        input_text = processor.apply_chat_template(conversation_history, add_generation_prompt=True)
        inputs = processor(
            None,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=500)
        des = processor.decode(output[0])
        generated_text = des.split("assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip("\n").replace("\n\n",
                                                                                                               " ")
        caption_dict[i] = generated_text
    address = os.path.join(args.save, args.data)
    with open(os.path.join(address, "caption_llava.pkl"), "wb") as f:
        pickle.dump(caption_dict, f)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default="")
    parser.add_argument('--data', default="ham")
    args = parser.parse_args()
    rank_caption(args)
