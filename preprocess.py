import torch
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle
import random

# device=torch.device('cuda:0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocesser = clip.load("ViT-B/32", device=device, jit=False)
CAPTION_PATH = './data/AllVideoDescriptions.txt'

# Method to get image, caption list which corresponds to a single video
# Each image is a 768 dimension vector
# Suppose a,b,c,d is 4 image corresponds to 1 video, this method returns a [a,b,c,d] vector which concate them together and a caption list.
def align_video_with_caption(video_name, all_captions, preprocesser):
    images = []
    for i in range(1,100):
        image_file = './images/'+video_name+'image'+str(i)+'.jpg'
        if not os.path.isfile(image_file):
            break
        image = preprocesser(Image.open(image_file)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_encoding = clip_model.encode_image(image).cpu()
        images.append(image_encoding)

    images = torch.cat(images, dim=0)
    captions = all_captions[video_name]
    return images, captions

# Brute Force: Simply convert video to 1 image frame (we will analyze it in our report why this makes sense).
def video_to_single_caption(video_name, all_captions, preprocesser):
    idx = random.randint(1,3)
    image_file = './images/'+video_name+'image'+str(idx)+'.jpg'

    if not os.path.isfile(image_file):
        idx = 1
        image_file = './images/' + video_name + 'image' + str(idx) + '.jpg'
        image = preprocesser(Image.open(image_file)).unsqueeze(0).to(device)

    image = preprocesser(Image.open(image_file)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_encoding = clip_model.encode_image(image).cpu()
    captions = all_captions[video_name]
    return image_encoding, captions

# The method to build a caption dictionary
def build_caption_dict():
    cap_dict = {}
    with open(CAPTION_PATH,'r') as f:
        # remove the headers
        ls = f.readlines()[7:]
        for l in tqdm(ls):
            l = l.strip().split()
            video_name=l[0]
            if video_name not in cap_dict:
                cap_dict[video_name] = []
            cap = l[1:]
            cap_dict[video_name].append(cap)
    return cap_dict

# The workflow to build our dataset
def build_dataset():
    all_images = []
    all_captions = []
    all_cap_dict = build_caption_dict()
    # One image per video
    video_single_images = []

    for video_name in tqdm(all_cap_dict):
        images, captions= align_video_with_caption(video_name,all_cap_dict,preprocesser)
        all_images.append(images)
        all_captions.append(captions)
        # Create (video -> single image, caption list) dataset
        single_image, captions = video_to_single_caption(video_name, all_cap_dict, preprocesser)
        video_single_images.append(single_image)

    assert len(all_images)==len(all_captions)

    with open('save_dataset.pkl', 'wb') as fw:
        pickle.dump((all_images, all_captions),fw)

    # The method to take means of all the images
    all_images_mean = torch.mean(torch.concat(all_images, dim=0), dim=0)
    with open('save_dataset_mean.pkl', 'wb') as fw:
        pickle.dump((all_images_mean, all_captions),fw)
    # Video -> Single image, Caption List dataset
    with open('save_dataset_single.pkl', 'wb') as fw:
        pickle.dump((video_single_images, all_captions), fw)


if __name__ == "__main__":
    build_dataset()
