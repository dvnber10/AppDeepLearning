# seq2seq_bahdanau.py
import re
import math
import random
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Preprocesamiento
# -----------------------------
def simple_tokenizer(text: str) -> List[str]:
    return re.findall(r"[a-záéíóúñü]+", str(text).lower())

class Vocab:
    def __init__(self, tokens_list, min_freq=1, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        self.freq = {}
        for tokens in tokens_list:
            for t in tokens:
                self.freq[t] = self.freq.get(t, 0) + 1
        # keep tokens with freq >= min_freq
        toks = [t for t, f in self.freq.items() if f >= min_freq]
        toks = sorted(toks)
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"] + reserved_tokens + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)

def build_dataset(file_path, src_col="english", trg_col="spanish", max_len=30):
    df = pd.read_csv(file_path)[[src_col, trg_col]].dropna().reset_index(drop=True)
    df["src_tok"] = df[src_col].apply(simple_tokenizer)
    df["trg_tok"] = df[trg_col].apply(simple_tokenizer)
    # add <sos> and <eos>
    df["trg_tok"] = df["trg_tok"].apply(lambda t: ["<sos>"] + t[: max_len-2] + ["<eos>"])
    df["src_tok"] = df["src_tok"].apply(lambda t: t[: max_len-2] + ["<eos>"])
    return df

# -----------------------------
# 2) Dataset and Collate
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, df, src_vocab, trg_vocab):
        self.src = df["src_tok"].tolist()
        self.trg = df["trg_tok"].tolist()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = [self.src_vocab.stoi.get(t, self.src_vocab.stoi["<unk>"]) for t in self.src[idx]]
        trg = [self.trg_vocab.stoi.get(t, self.trg_vocab.stoi["<unk>"]) for t in self.trg[idx]]
        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    trg_lens = [len(t) for t in trg_batch]
    src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_pad = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_pad, torch.tensor(src_lens), trg_pad, torch.tensor(trg_lens)

# -----------------------------
# 3) Model components
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [B, L, E]
        # pack
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)  # hidden: [n_layers, B, hid_dim]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, L, hid_dim]
        return out, hidden  # out for attention, hidden for decoder init

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.W2 = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.V = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden, mask=None):
        # encoder_outputs: [B, src_len, enc_hid]
        # hidden: [n_layers, B, dec_hid] -> use last layer
        dec_hidden = hidden[-1].unsqueeze(1)  # [B,1,dec_hid]
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(dec_hidden)))  # [B, src_len, 1]
        score = score.squeeze(-1)  # [B, src_len]
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(score, dim=1)  # [B, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B,1,enc_hid]
        context = context.squeeze(1)  # [B, enc_hid]
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.attention = attention
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, encoder_outputs, mask=None):
        # input_step: [B] (tokens at current timestep)
        embedded = self.dropout(self.embedding(input_step).unsqueeze(1))  # [B,1,E]
        context, attn_weights = self.attention(encoder_outputs, hidden, mask)  # context: [B, enc_hid]
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)  # [B,1, E+enc_hid]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [B,1,hid], hidden: [n_layers,B,hid]
        output = output.squeeze(1)  # [B, hid]
        # concat for final prediction
        pred_input = torch.cat([output, context, embedded.squeeze(1)], dim=1)  # [B, hid+enc_hid+emb]
        pred = self.fc_out(pred_input)  # [B, output_dim]
        return pred, hidden, attn_weights

# -----------------------------
# 4) Training & evaluation utils
# -----------------------------
def create_mask(src_pad):
    # src_pad: [B, src_len]
    return (src_pad != 0).to(src_pad.device)  # 1 where not pad

def train_epoch(encoder, decoder, dataloader, encoder_opt, decoder_opt, criterion, device, clip=1.0, teacher_forcing_ratio=0.5):
    encoder.train(); decoder.train()
    epoch_loss = 0
    for src_pad, src_lens, trg_pad, trg_lens in dataloader:
        src_pad = src_pad.to(device); trg_pad = trg_pad.to(device); src_lens = src_lens.to(device)
        batch_size, trg_len = trg_pad.size()
        encoder_opt.zero_grad(); decoder_opt.zero_grad()
        encoder_outputs, hidden = encoder(src_pad, src_lens)  # encoder_outputs: [B,src_len,enc_hid]
        mask = create_mask(src_pad)

        # first input to decoder is <sos> tokens
        input_tok = trg_pad[:, 0]  # [B]
        loss = 0
        for t in range(1, trg_len):
            output, hidden, _ = decoder(input_tok, hidden, encoder_outputs, mask)
            # output: [B, out_dim] (logits)
            loss_step = criterion(output, trg_pad[:, t])
            loss += loss_step
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = trg_pad[:, t] if teacher_force else top1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_opt.step(); decoder_opt.step()
        epoch_loss += loss.item() / (trg_len - 1)
    return epoch_loss / len(dataloader)

def evaluate(encoder, decoder, dataloader, criterion, device):
    encoder.eval(); decoder.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src_pad, src_lens, trg_pad, trg_lens in dataloader:
            src_pad = src_pad.to(device); trg_pad = trg_pad.to(device); src_lens = src_lens.to(device)
            encoder_outputs, hidden = encoder(src_pad, src_lens)
            mask = create_mask(src_pad)
            input_tok = trg_pad[:, 0]
            loss = 0
            trg_len = trg_pad.size(1)
            for t in range(1, trg_len):
                output, hidden, _ = decoder(input_tok, hidden, encoder_outputs, mask)
                loss += criterion(output, trg_pad[:, t])
                top1 = output.argmax(1)
                input_tok = top1
            epoch_loss += loss.item() / (trg_len - 1)
    return epoch_loss / len(dataloader)

# -----------------------------
# 5) Inference (greedy)
# -----------------------------
def translate_sentence(sentence_tokens, src_vocab, trg_vocab, encoder, decoder, device, max_len=40):
    encoder.eval(); decoder.eval()
    src_idx = [src_vocab.stoi.get(t, src_vocab.stoi["<unk>"]) for t in sentence_tokens]
    src_tensor = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0).to(device)
    src_len = torch.tensor([len(src_idx)]).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = encoder(src_tensor, src_len)
        mask = create_mask(src_tensor)
        input_tok = torch.tensor([trg_vocab.stoi["<sos>"]], dtype=torch.long).to(device)
        result_tokens = []
        for _ in range(max_len):
            output, hidden, attn = decoder(input_tok, hidden, encoder_outputs, mask)
            top1 = output.argmax(1).item()
            if top1 == trg_vocab.stoi["<eos>"]:
                break
            result_tokens.append(trg_vocab.itos[top1])
            input_tok = torch.tensor([top1], dtype=torch.long).to(device)
    return result_tokens

# -----------------------------
# 6) Main routine
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print("Device:", device)

    # Load and preprocess
    df = build_dataset(args.file_path, max_len=args.max_len)
    # Build vocabs
    src_vocab = Vocab(df["src_tok"].tolist(), min_freq=args.min_freq)
    trg_vocab = Vocab(df["trg_tok"].tolist(), min_freq=args.min_freq)

    print("Vocab sizes -> src:", len(src_vocab), " trg:", len(trg_vocab))
    # Split
    df_shuf = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df_shuf) * (1 - args.val_ratio))
    df_train = df_shuf[:n_train]
    df_val = df_shuf[n_train:]

    train_ds = TranslationDataset(df_train, src_vocab, trg_vocab)
    val_ds = TranslationDataset(df_val, src_vocab, trg_vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    encoder = Encoder(len(src_vocab), args.emb_dim, args.hid_dim, n_layers=args.n_layers, dropout=args.dropout).to(device)
    attention = BahdanauAttention(args.hid_dim, args.hid_dim).to(device)
    decoder = Decoder(len(trg_vocab), args.emb_dim, args.hid_dim, attention, n_layers=args.n_layers, dropout=args.dropout).to(device)

    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(encoder, decoder, train_loader, encoder_opt, decoder_opt, criterion, device, clip=args.clip, teacher_forcing_ratio=args.teacher_forcing)
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "src_vocab": {
                    "itos": src_vocab.itos,
                    "stoi": src_vocab.stoi
                },
                "trg_vocab": {
                    "itos": trg_vocab.itos,
                    "stoi": trg_vocab.stoi
                },
                "model_config": {
                    "emb_dim": args.emb_dim,
                    "hid_dim": args.hid_dim,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout
                }
                
            }, args.save_path)
            print("Modelo guardado en", args.save_path)
            

    # Demo: traduzca algunas frases del set de validación
    for i in range(min(10, len(df_val))):
        toks = df_val.iloc[i]["src_tok"]
        pred = translate_sentence(toks, src_vocab, trg_vocab, encoder, decoder, device, max_len=args.max_len)
        print("SRC:", " ".join(toks))
        print("PRED:", " ".join(pred))
        print("TRG :", " ".join(df_val.iloc[i]["trg_tok"][1:-1]))  # sin <sos> <eos>
        print("---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="dataset.csv")
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--teacher_forcing", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="best_seq2seq.pt")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    main(args)
