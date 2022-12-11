import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import numpy as np


T = torch.Tensor

TN = Optional[T]




D = torch.device
CPU = torch.device('cpu')

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        pad_mask = self.pad_masks[self.caption2embedding[item]]
        '''
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        '''
        return tokens, mask, prefix, pad_mask

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        self.pad_masks = all_data["masks"]
        captions_raw = all_data["captions"]
        #self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import math, copy, time

from torch.autograd import Variable


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ManualEncoderLayer(nn.Module):
    def __init__(self, dim_emb, dropout, nhead, dim_ff):
        super(ManualEncoderLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(dim_emb, nhead)
        self.ffn = PositionwiseFeedForward(dim_emb, dim_ff)
        self.dim_emb = dim_emb
        self.sub_layer1 = SublayerConnection(dim_emb, dropout)
        self.sub_layer2 = SublayerConnection(dim_emb, dropout)
        self.dim_ff = dim_ff


    def forward(self, x, src_mask, padding_mask):
        x = self.sub_layer1(x, lambda x: self.attention(x, x, x, attn_mask=src_mask, key_padding_mask=padding_mask)[0])
        x = self.sub_layer2(x, self.ffn)
        return x
class ManualEncoder(nn.Module):
    def __init__(self, layer, N):
        super(ManualEncoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = nn.LayerNorm(layer.dim_emb)
        #TODO: Initialize the necessary pieces of the encoder
        # (Hint, the mostly consists of making copies of your encoder layers)

    def forward(self, x, src_mask, padding_mask):
        for mod in self.layers:
          x = mod(x, src_mask, padding_mask)
        x = self.norm(x)
        return x




class ManualTransformer(nn.Module):
    def __init__(self, num_encoder_layers, emb_size, nhead=8, dim_feedforward=512, dropout=0.1):
        super(ManualTransformer, self).__init__()
        self.encoder = ManualEncoder(ManualEncoderLayer(emb_size, dropout, nhead, dim_feedforward), num_encoder_layers)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_src_mask(self, src):
        return torch.zeros((src.shape[0], src.shape[0]), device='cuda:0').type(torch.bool)

    def encode(self, src, src_padding_mask):
        src = self.positional_encoding(src)
        src_mask = self.get_src_mask(src)
        return self.encoder(src, src_mask, src_padding_mask)

    def forward(self, src, src_padding_mask):
        memory = self.encode(src, src_padding_mask)
        return memory
class TransformerMapper(nn.Module):
    def __init__(self, dim_clip, clip_length, dim_embedding,num_encoder_layers):
        super(TransformerMapper, self).__init__()
        self.linear = nn.Linear(dim_clip,dim_embedding*clip_length)
        self.transformer = ManualTransformer(num_encoder_layers, dim_clip)
        self.indices = torch.tensor([0]).to('cuda:0')
        self.cls_encoding = nn.Parameter(torch.randn(1,1,dim_clip), requires_grad=True)
    def forward(self,x,padding_mask):
        x = torch.cat((self.cls_encoding.repeat(1,x.shape[1],1),x),dim=0)
        encoding = self.transformer(x,padding_mask)
        encoding = torch.transpose(encoding,dim0=0,dim1=1)
        cls = torch.index_select(encoding, 1, self.indices)
        output = self.linear(cls)
        return output






class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, pad_mask: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix = torch.transpose(prefix,0,1)
        prefix_projections = self.clip_project(prefix,pad_mask).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, num_layers: int,prefix_size: int = 512,):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.clip_project = TransformerMapper(dim_clip=prefix_size, dim_embedding=self.gpt_embedding_size,
                                                  clip_length=prefix_length, num_encoder_layers=num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)





def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix,pad_mask) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), pad_mask.to(device)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask, pad_mask=pad_mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='save_dataset_seq.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='msv_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true',default=True)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    model = ClipCaptionPrefix(prefix_length=prefix_length, prefix_size=prefix_dim,
                                  num_layers=args.num_layers)
    print("Train only prefix")


    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

if __name__ == '__main__':
    main()
