from typing import Iterable, List

import torch
import torchdata.datapipes as dp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE = 'eng'
TGT_LANGUAGE = 'zh'
file_path = 'data/eng-zh.txt'
val_file = "data/val.txt"
batch_size = 64
#
text_transform = {}
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_sm')
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

train_pipe = dp.iter.IterableWrapper([file_path])
train_pipe = dp.iter.FileOpener(train_pipe, mode='rb')
train_pipe = train_pipe.parse_csv(skip_lines=0, delimiter='\t')

val_pipe = dp.iter.IterableWrapper([val_file])
val_pipe = dp.iter.FileOpener(val_pipe, mode='rb')
val_pipe = val_pipe.parse_csv(skip_lines=0, delimiter='\t')


def removeAttribution(row):
    return row[:2]


# 删除行多余数据
train_pipe = train_pipe.map(removeAttribution)
# 删除行多余数据
val_pipe = val_pipe.map(removeAttribution)


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_pipe, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               tensor_transform)  # Add BOS/EOS and create tensor


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


train_dataloader = DataLoader(train_pipe, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
# ================================
val_dataloader = DataLoader(val_pipe, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
