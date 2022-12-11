import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from IPython.display import Image

from train_seq import ClipCaptionModel, ClipCaptionPrefix

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


torch.random.manual_seed(333)
padding_embdding = torch.randn(512).unsqueeze(0).to('cuda:0')
# Beam search on tokens to generate sequence of tokens
def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def main():

    device = 'cuda:0'
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10

    model = ClipCaptionModel(prefix_length=prefix_length, prefix_size=512,
                                  num_layers=6)
    model_path = 'msv_train/msv_prefix-099.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model = model.eval()
    model = model.to(device)
    image_list=['HkpUWzNNVt4_20_30','RMznbCn5sQs_0_10','aM-RcQj0a7I_37_55','R8FDJgVW3Vc_0_4']
    use_beam_search = False # @param {type:"boolean"}

    def align_video_with_caption(video_name, all_captions, preprocesser, padding=20, pad=False):
        images = []
        masks = [0]
        for i in range(1, padding + 1):
            image_file = '../vit/images/' + video_name + 'image' + str(i) + '.jpg'
            if not os.path.isfile(image_file):
                break
            image = preprocesser(PIL.Image.open(image_file)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_encoding = clip_model.encode_image(image)
            images.append(image_encoding)
            masks.append(0)
        if pad:
            while len(images) < padding:
                images.append(padding_embdding)
                masks.append(1)
        images = torch.cat(images, dim=0).unsqueeze(0).to(device)
        captions = all_captions[video_name]
        masks = torch.Tensor(masks).type(torch.bool).unsqueeze(0).to(device)
        return images, captions, masks

    def build_caption_dict():
        cap_dict = {}
        with open('../vit/AllVideoDescriptions.txt', 'r') as f:
            # remove the headers
            ls = f.readlines()[7:]
            for l in tqdm(ls):
                l = l.strip().split()
                video_name = l[0]
                if video_name not in cap_dict:
                    cap_dict[video_name] = []
                cap = ' '.join(l[1:]) + '.'
                cap_dict[video_name].append(cap)
        return cap_dict
    all_cap_dict=build_caption_dict()
    i=0
    for video_name in image_list:
        images, captions,masks = align_video_with_caption(video_name, all_cap_dict, preprocess,pad=True)
        #pil_img = Image(filename=UPLOADED_FILE)

        with torch.no_grad():
            # if type(model) is ClipCaptionE2E:
            #     prefix_embed = model.forward_image(image)
            # else:
            model = model.to(device)
            prefix = torch.transpose(images,0,1)
            prefix.to(device)
            masks.to(device)
            prefix_embed = model.clip_project(prefix,masks).reshape(1, prefix_length, -1)

        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)


        print('\n')
        print(video_name)
        print(captions[0])
        print(generated_text_prefix)
        i+=1
        if i>10:
            break

if __name__ == '__main__':
    main()
