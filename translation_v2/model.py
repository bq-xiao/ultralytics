import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer, Module
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import PAD_IDX, batch_size, BOS_IDX, EOS_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


class Seq2SeqModule(pl.LightningModule):

    def __init__(self, transformer: Module, loss, learning_rate=2e-5):
        super().__init__()
        self.transformer = transformer
        self.loss = loss
        self.lr = learning_rate
        self.plot_losses = []

    def forward(self, x):
        x = x.to(DEVICE)
        src = x.view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        src_mask = src_mask.to(DEVICE)
        memory = self.transformer.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
        max_len = num_tokens + 5
        for i in range(max_len - 1):
            memory = memory.to(DEVICE)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
            out = self.transformer.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys.flatten()

    def compute_loss(self, y_hat, y):
        y = y[1:, :]
        return self.loss(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        tgt_input = y[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(x, tgt_input)
        x_hat = self.transformer(x, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        loss = self.compute_loss(x_hat, y)
        self.plot_losses.append(loss)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        tgt_input = y[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(x, tgt_input)
        x_hat = self.transformer(x, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        test_loss = self.compute_loss(x_hat, y)

        self.log('test_loss', test_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        return test_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        tgt_input = y[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(x, tgt_input)
        x_hat = self.transformer(x, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        val_loss = self.compute_loss(x_hat, y)

        self.log('val_loss', val_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3,
                                               eps=1e-9, verbose=True),
                "interval": "epoch",
                "monitor": "train_loss",
                "frequency": 1
            },
        }

    def get_plot_losses(self):
        return self.plot_losses
