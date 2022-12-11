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
CAPTION_PATH = '../vit/AllVideoDescriptions.txt'
torch.random.manual_seed(333)
padding_embdding = torch.randn(512).unsqueeze(0)
# Method to get image, caption list which corresponds to a single video
# Each image is a 768 dimension vector
# Suppose a,b,c,d is 4 image corresponds to 1 video, this method returns a [a,b,c,d] vector which concate them together and a caption list.
def align_video_with_caption(video_name, all_captions, preprocesser,padding=20,pad=False):
    images = []
    masks = [0]
    for i in range(1,padding+1):
        image_file = '../vit/images/'+video_name+'image'+str(i)+'.jpg'
        if not os.path.isfile(image_file):
            break
        image = preprocesser(Image.open(image_file)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_encoding = clip_model.encode_image(image).cpu()
        images.append(image_encoding)
        masks.append(0)
    if pad:
        while len(images)<padding:
            images.append(padding_embdding)
            masks.append(1)
    images = torch.cat(images, dim=0).unsqueeze(0)
    captions = all_captions[video_name]
    masks =torch.Tensor(masks).type(torch.bool).unsqueeze(0)
    return images, captions,masks

# Brute Force: Simply convert video to 1 image frame (we will analyze it in our report why this makes sense).
def video_to_single_caption(video_name, all_captions, preprocesser):
    idx = random.randint(1,3)
    image_file = '../vit/images/'+video_name+'image'+str(idx)+'.jpg'

    if not os.path.isfile(image_file):
        idx = 1
        image_file = '../vit/images/' + video_name + 'image' + str(idx) + '.jpg'
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
            cap = ' '.join(l[1:])+'.'
            cap_dict[video_name].append(cap)
    return cap_dict

# The workflow to build our dataset
def build_dataset():
    all_images = []
    all_captions = []
    all_cap_dict = build_caption_dict()
    # One image per video
    video_single_images = []
    mean_images = []
    i=0
    all_video_names = []
    for video_name in tqdm(all_cap_dict):
        #images, captions,_= align_video_with_caption(video_name,all_cap_dict,preprocesser)
        #all_images.append(images)
        #all_captions.append(captions)
        # Create (video -> single image, caption list) dataset

        single_image, captions = video_to_single_caption(video_name, all_cap_dict, preprocesser)
        video_single_images.append(single_image)
        d={}
        d["clip_embedding"] = i
        i += 1
        d["caption"] = captions
        all_captions.append(d)
        print(d)
        #mean_image = torch.mean(images,dim=0)
        #mean_images.append(mean_image.unsqueeze(0))
        all_video_names.append(video_name)
    with open('save_dataset_sibgle.pkl', 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(video_single_images, dim=0), "captions": all_captions,"video_names":all_video_names}, f)
def build_mean_dataset():
    all_images = []
    all_captions = []
    all_cap_dict = build_caption_dict()
    # One image per video
    video_single_images = []
    mean_images = []
    i=0
    all_video_names = []
    for video_name in tqdm(all_cap_dict):
        images, captions,_= align_video_with_caption(video_name,all_cap_dict,preprocesser)
        #all_images.append(images)
        #all_captions.append(captions)
        # Create (video -> single image, caption list) dataset

        #single_image, caption = video_to_single_caption(video_name, all_cap_dict, preprocesser)
        #video_single_images.append(single_image)
        d={}
        d["clip_embedding"] = i
        i += 1
        d["caption"] = captions
        all_captions.append(d)
        print(d)
        mean_image = torch.mean(images,dim=0)
        mean_images.append(mean_image.unsqueeze(0))
        all_video_names.append(video_name)
    with open('save_dataset_mean.pkl', 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(mean_images, dim=0), "captions": all_captions,"video_names":all_video_names}, f)
def build_seq_dataset():
    all_images = []
    all_captions = []
    all_cap_dict = build_caption_dict()
    i = 0
    all_masks = []
    all_video_names = []
    for video_name in tqdm(all_cap_dict):
        images, captions, masks= align_video_with_caption(video_name,all_cap_dict,preprocesser,padding=20,pad=True)
        all_images.append(images)
        all_masks.append(masks)
        d = {}
        d["clip_embedding"] = i
        i += 1
        d["caption"] = captions
        all_captions.append(d)
        all_video_names.append(video_name)
    all_images=torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    print(all_images.shape)
    print(all_masks.shape)
    with open('save_dataset_seq.pkl', 'wb') as f:
        pickle.dump({"clip_embedding": all_images, "captions": all_captions, "masks": all_masks,"video_names":all_video_names}, f)
if __name__ == "__main__":
    #build_seq_dataset()
    build_dataset()
