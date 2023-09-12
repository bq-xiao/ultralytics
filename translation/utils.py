import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class CustomTextDataset(Dataset):
    def __init__(self, data_pipe):
        self.data_pipe = data_pipe
        self.data = list(data_pipe)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_text_dataset(file_path='data/mock.txt', batch_size=32, max_length=128, num_workers=0):
    tokenize_zh = get_tokenizer('spacy', language='zh_core_web_sm')
    tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')

    data_pipe = dp.iter.IterableWrapper([file_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t')

    def removeAttribution(row):
        return row[:2]

    # 删除行多余数据
    data_pipe = data_pipe.map(removeAttribution)

    def getTokens(data_iter, place):
        for english, german in data_iter:
            if place == 0:
                yield tokenize_en(english)
            else:
                yield tokenize_zh(german)

    en_vocab = build_vocab_from_iterator(
        getTokens(data_pipe, 0),
        min_freq=2,
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )
    en_vocab.set_default_index(en_vocab['<unk>'])

    zh_vocab = build_vocab_from_iterator(
        getTokens(data_pipe, 1),
        min_freq=2,
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )
    zh_vocab.set_default_index(zh_vocab['<unk>'])

    def getTransform(vocab):
        text_tranform = T.Sequential(
            ## converts the sentences to indices based on given vocabulary
            T.VocabTransform(vocab=vocab),
            ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
            # 1 as seen in previous section
            T.AddToken(1, begin=True),
            ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
            # 2 as seen in previous section
            T.AddToken(2, begin=False)
        )
        return text_tranform

    def applyTransform(sequence_pair):
        return (getTransform(en_vocab)(tokenize_en(sequence_pair[0])),
                getTransform(zh_vocab)(tokenize_zh(sequence_pair[1])))

    # 构建字典
    data_pipe = data_pipe.map(applyTransform)

    def applyPadding(pair_of_sequences):
        text_tranform = T.Sequential(
            T.ToTensor(0),
            T.PadTransform(max_length=max_length, pad_value=0)
        )
        return (text_tranform(list(pair_of_sequences[0])), text_tranform(list(pair_of_sequences[1])))

    data_pipe = data_pipe.map(applyPadding)
    source_index_to_string = en_vocab.get_itos()
    target_index_to_string = zh_vocab.get_itos()

    def showSomeTransformedSentences(data_loader):
        """
        Function to show how the sentences look like after applying all transforms.
        Here we try to print actual words instead of corresponding index
        """
        n = len(data_loader)
        print(f"n size: {n}")
        for sources, targets in data_loader:
            sources = torch.transpose(sources, 0, 1)
            targets = torch.transpose(targets, 0, 1)
            print(f"sources size: {sources.shape} \t targets size: {targets.shape}")
            if sources[0][-1] != 0:
                continue  # Just to visualize padding of shorter sentences
            for i in range(n):
                source = ""
                for token in sources[i]:
                    source += " " + source_index_to_string[token]
                target = ""
                for token in targets[i]:
                    target += " " + target_index_to_string[token]
                print(f"Source: {source}")
                print(f"Traget: {target}")

    # train_iter = DataLoader(dataset=data_pipe, batch_size=batch_size, num_workers=0)
    datasets = CustomTextDataset(data_pipe)
    full_size = len(datasets)
    train_size = int(0.8 * full_size)
    test_size = full_size - train_size
    train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

    validate_size = int(0.5 * test_size)
    netxt_size = test_size - validate_size
    test_dataset, val_dataset = random_split(test_dataset, [validate_size, netxt_size])
    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}")
    print(f"val_dataset: {len(val_dataset)}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)
    # showSomeTransformedSentences(train_loader)

    return train_loader, test_loader, val_loader, en_vocab, zh_vocab

# load_text_dataset(batch_size=8)
