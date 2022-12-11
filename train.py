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
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        caption = self.captions[item]
        video = self.videos[item]
        return tokens, mask, prefix,caption,video


# The Multi Layer Perceptron part of the Transformer
class MLPTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim: Optional[int] = None, activation=F.relu, dropout=0.):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# A MultiHeadAttention Module
class MultiHeadAttention(nn.Module):

    def __init__(self, self_dim, target_dim, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = self_dim // num_heads
        self.scale = head_dim ** -0.5
        self.x2query = nn.Linear(self_dim, self_dim, bias=bias)
        self.y2keys_values = nn.Linear(target_dim, self_dim * 2, bias=bias)
        self.fc = nn.Linear(self_dim, self_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x
        # b - batch size(# sentences), n - len(x_sentenceA), length of a single sentence, c - len(words)
        b, n, c = x.shape

        # m - len(y_sentenceA), d = len(single words)
        _, m, d = y.shape
        # b n h dh
        queries = self.x2query(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.y2keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        # https://pytorch.org/docs/stable/generated/torch.einsum.html
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.fc(out)
        return out, attention

# Transformer + MLPTransformer
class TransformerLayer(nn.Module):

    def __init__(self, self_dim, target_dim, num_heads, mlp_ratio=4., bias=False, dropout=0., activation=F.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(self_dim)
        self.attn = MultiHeadAttention(self_dim, target_dim, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(self_dim)

        self.mlp = MLPTransformer(self_dim, int(self_dim * mlp_ratio), activation=activation, dropout=dropout)

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_with_attention(self, x, y=None, mask=None):
        z, attention = self.attn(self.norm1(x), y, mask)
        x = x + z
        x = x + self.mlp(self.norm2(x))
        return x, attention

# Transformer model
class Transformer(nn.Module):
    def __init__(self, self_dim: int, num_heads: int, num_layers: int, target_dim: Optional[int] = None,
                 mlp_ratio: float = 2., activation=F.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        if target_dim is None:
            target_dim = self_dim
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            # cross
            if i % 2 == 0 and enc_dec:
                layers.append(TransformerLayer(self_dim, target_dim, num_heads, mlp_ratio, activation=activation, norm_layer=norm_layer))
            # self
            elif enc_dec:
                layers.append(TransformerLayer(self_dim, self_dim, num_heads, mlp_ratio, activation=activation, norm_layer=norm_layer))
            # self or cross
            else:
                layers.append(TransformerLayer(self_dim, target_dim, num_heads, mlp_ratio, activation=activation, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            # cross
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y)
            # self
            elif self.enc_dec:
                x = layer(x, x, mask)
            # self or cross
            else:
                x = layer(x, y, mask)
        return x

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, attn = layer.forward_with_attention(x, y, mask)
            attentions.append(attn)
        return x, attentions

# A transformer mapper
class TransformerMapper(nn.Module):
    def __init__(self, clip_dim: int, embed_dim: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(embed_dim, 8, num_layers)
        # clip_dim -> clip_length * embed_dim
        self.fc = nn.Linear(clip_dim, clip_length * embed_dim)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, embed_dim), requires_grad=True)

    def forward(self, x):
        x = self.fc(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], * self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out # out transformer(video prefix)

# A series of MLP
class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, activation=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, clip_length: int, num_layers: int,
                 prefix_size: int = 512, ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # get embedding for the captions
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]


        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)

    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
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
        for idx, (tokens, mask, prefix,_,_) in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask)
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
        for idx, (tokens, mask, prefix,_,_) in enumerate(val_dataloader):
            model.eval()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask)
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
    for tokens, mask, prefix,  caption,video in test_set:
        tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device,dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
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
    with open('save_test_data_mean.pkl','wb') as fwb:
        pickle.dump(all_data,fwb)







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='save_dataset_mean.pkl')
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
                              num_layers=args.num_layers,clip_length=prefix_length)
    print("Train only prefix")
    torch.random.manual_seed(333)
    train_size = int(len(dataset) * 0.9)

    all_train, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_size = int(len(all_train) * 0.9)
    train_set, val_set = torch.utils.data.random_split(all_train, [train_size, len(all_train) - train_size])
    train(train_set,val_set, model, args, output_dir=args.out_dir, output_prefix=args.prefix,prefix_length=prefix_length)
    model = ClipCaptionModel(prefix_length=prefix_length, prefix_size=prefix_dim,
                              num_layers=args.num_layers,clip_length=prefix_length)
    model_path = 'msv_train/msv_prefix-49.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test(model,test_set,prefix_length,tokenizer)





if __name__ == '__main__':
    main()