import argparse
import math

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from model import Encoder, Decoder, Seq2Seq
from translation.utils import load_text_dataset


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=16,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, en_vocab):
    with torch.no_grad():
        model.eval()
        pad = en_vocab['<pad>']
        total_loss = 0
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            src = src.data.cuda()
            trg = trg.data.cuda()
            output = model(src, trg, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                              trg[1:].contiguous().view(-1),
                              ignore_index=pad)
            total_loss += loss.data.item()
        return total_loss / len(val_iter)


def train(model, optimizer, train_iter, vocab_size, grad_clip, en_vocab):
    model.train()
    total_loss = 0
    pad = en_vocab['<pad>']
    for n, batch in enumerate(train_iter):
        src, trg = batch
        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        # print(f"src:{src.shape}, trg:{trg.shape}")
        output = model(src, trg)
        # print(f"output:{output.shape}")
        # a = output[1:]
        # print(f"a:{a.shape}")
        # b = a.view(-1, vocab_size)
        # print(f"b:{b.shape}")
        # c = trg[1:]
        # print(f"c:{c.shape}")
        # d = c.contiguous()
        # print(f"d:{d.shape}")
        # e = d.view(-1)
        # print(f"e:{e.shape}")
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()
        # print(f"train loss:{loss.data.item()}")
        if n % 50 == 0 and n != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (n, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_loader, test_loader, val_loader, en_vocab, zh_vocab = \
        load_text_dataset(file_path='data/eng-zh.txt', batch_size=args.batch_size, num_workers=0)
    en_size, zh_size = len(en_vocab), len(zh_vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)" % (len(train_loader), len(train_loader.dataset),
                                                               len(test_loader), len(test_loader.dataset)))
    print("[en_vocab]:%d [zh_vocab]:%d" % (en_size, zh_size))

    print("[!] Instantiating models...")
    encoder = Encoder(en_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(seq2seq, optimizer, train_loader, zh_size, args.grad_clip, en_vocab)
        val_loss = evaluate(seq2seq, val_loader, zh_size, en_vocab)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS" % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            torch.save(seq2seq.state_dict(), 'seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_loader, en_size, en_vocab)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
