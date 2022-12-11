import torch
import torch.nn as nn
import os
import pickle
import sys
import argparse
import numpy as np
import json
from typing import Tuple, Optional, Union
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn import functional as nnf
T = torch.Tensor
TN = Optional[T]
D = torch.device
CPU = torch.device('cpu')

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

# Deal with Captions and embed it as tokens, mask, prefix
# Use GPT2Tokenizer to do the tokenization and operate padding, mask on tokens
class ClipCocoDataset(Dataset):
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
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
        # self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        self.videos = all_data['video_names']

        # For debugging
        #print(self.captions[:10])

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                # tokenize the captions
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption'][0]), dtype=torch.int64))
                # turn captions into embeddings
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

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
        # mask is zero where we out of sequence
        # https://pytorch.org/docs/stable/generated/torch.ge.html
        # > 0, true
        mask = tokens.ge(0)
        tokens[~mask] = 0
        # adding prefix mask
        mask = torch.cat((torch.ones(self.prefix_length), mask.float()), dim=0)
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        pad_mask = self.pad_masks[self.caption2embedding[item]]
        caption =self.captions[item]
        video = self.videos[item]
        # if self.normalize_prefix:
        #     prefix = prefix.float()
        #     prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix, pad_mask, caption,video

# Turn token into embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Add a positional Encoding
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

# Add sublayer Connection
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Positionwised feed forward layer
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x).relu()))

# A method to clone N
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# A manual implementation of encoder layer
class ManualEncoderLayer(nn.Module):
    def __init__(self, emb_dim, dropout, nhead, ff_dim):
        super(ManualEncoderLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(emb_dim, nhead)
        self.ffn = PositionwiseFeedForward(emb_dim, ff_dim)
        self.emb_dim = emb_dim
        self.sub_layer1 = SublayerConnection(emb_dim, dropout)
        self.sub_layer2 = SublayerConnection(emb_dim, dropout)
        self.dim_ff = ff_dim

    def forward(self, x, src_mask, padding_mask):
        x = self.sub_layer1(x, lambda x: self.attention(x, x, x, attn_mask=src_mask, key_padding_mask=padding_mask)[0])
        x = self.sub_layer2(x, self.ffn)
        return x

# A manual implementation of encoder
class ManualEncoder(nn.Module):
    def __init__(self, layer, N):
        super(ManualEncoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = nn.LayerNorm(layer.emb_dim)

    def forward(self, x, src_mask, padding_mask):
        for mod in self.layers:
          x = mod(x, src_mask, padding_mask)
        x = self.norm(x)
        return x

# A manual implementation of transformer
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


# A transformer mapper
class TransformerMapper(nn.Module):
    def __init__(self, clip_dim, clip_length, embed_dim,num_encoder_layers):
        super(TransformerMapper, self).__init__()
        # clip_dim -> clip_length * embed_dim
        self.fc = nn.Linear(clip_dim , embed_dim * clip_length)

        self.transformer = ManualTransformer(num_encoder_layers, clip_dim)
        self.indices = torch.tensor([0]).to('cuda:0')
        self.cls_encoding = nn.Parameter(torch.randn(1,1,clip_dim), requires_grad=True)

    def forward(self,x,padding_mask):
        x = torch.cat((self.cls_encoding.repeat(1,x.shape[1],1),x),dim=0)
        encoding = self.transformer(x,padding_mask)
        encoding = torch.transpose(encoding,dim0=0,dim1=1)
        cls = torch.index_select(encoding, 1, self.indices)
        output = self.fc(cls)
        return output


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, num_layers: int,
                 prefix_size: int = 512, ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # get embedding for the captions
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.clip_project = TransformerMapper(clip_dim=prefix_size, embed_dim=self.gpt_embedding_size,
                                              clip_length=prefix_length, num_encoder_layers=num_layers)

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





def train(train_set,val_set, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",prefix_length=10):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix,pad_mask,_,_) in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), pad_mask.to(device)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask, pad_mask=pad_mask)
            logits = outputs.logits[:, prefix_length - 1: -1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
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
        progress2 = tqdm(total=len(val_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix,pad_mask,_,_) in enumerate(val_dataloader):
            model.eval()
            tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), pad_mask.to(device)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask, pad_mask=pad_mask)
            logits = outputs.logits[:, prefix_length - 1: -1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            progress2.set_postfix({"val_loss": loss.item()})
            progress2.update()

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress2.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model
def test(model,test_set,prefix_length,tokenizer):
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
            for entry_idx in range(entry_count):
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
    device='cuda:0'
    model = model.eval()
    model = model.to(device)
    use_beam_search = False
    fw=open('outputs.txt','w')
    all_data=[]
    id=0
    for tokens, mask, prefix, pad_mask, caption,video in test_set:
        tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device,dtype=torch.float32), pad_mask.to(device)
        prefix = prefix.unsqueeze(0)
        prefix = torch.transpose(prefix, 0, 1)
        pad_mask = pad_mask.unsqueeze(0)
        prefix_embed = model.clip_project(prefix, pad_mask).reshape(1, prefix_length, -1)
        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        '''
        TODO: eval(generated_text_prefix,caption)
        '''
        print(caption[0])
        print(generated_text_prefix)
        all_data.append({'video_name':video,'video_id':id,'pred_sentence':generated_text_prefix,'ref_sentences':caption})
    with open('save_test_data_seq.pkl','wb') as fwb:
        pickle.dump(all_data,fwb)







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
    torch.random.manual_seed(333)
    train_size = int(len(dataset) * 0.9)

    all_train, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_size = int(len(all_train) * 0.9)
    train_set, val_set = torch.utils.data.random_split(all_train, [train_size, len(all_train) - train_size])
    #train(train_set,val_set, model, args, output_dir=args.out_dir, output_prefix=args.prefix,prefix_length=prefix_length)
    model = ClipCaptionModel(prefix_length=prefix_length, prefix_size=prefix_dim,
                              num_layers=args.num_layers)
    model_path = 'msv_train/msv_prefix-049.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test(model,test_set,prefix_length,tokenizer)





if __name__ == '__main__':
    main()
