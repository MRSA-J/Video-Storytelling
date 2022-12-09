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
device=torch.device('cuda:0')
clip_model, preprocesser = clip.load("ViT-B/32", device=device, jit=False)
def align_video_with_caption(video_name,all_captions,preprocesser):
    images=[]
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
    return images,captions
def build_caption_dict():
    cap_dict={}
    with open('AllVideoDescriptions.txt','r') as f:
        ls = f.readlines()
        for l in tqdm(ls):
            l=l.strip().split()
            video_name=l[0]
            if video_name not in cap_dict:
                cap_dict[video_name]=[]
            cap = l[1:]
            cap_dict[video_name].append(cap)
    return cap_dict
def build_dataset():
    all_images=[]
    all_captions=[]
    all_cap_dict=build_caption_dict()
    for video_name in tqdm(all_cap_dict):
        images,captions=align_video_with_caption(video_name,all_cap_dict,preprocesser)
        all_images.append(images)
        all_captions.append(captions)
    assert len(all_images)==len(all_captions)
    with open('save_dataset.pkl','wb') as fw:
        pickle.dump((all_images,all_captions),fw)
build_dataset()




