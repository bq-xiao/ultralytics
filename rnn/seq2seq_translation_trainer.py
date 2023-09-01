import lightning.pytorch as pl
import spacy
import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from matplotlib import pyplot as plt, ticker
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator

from rnn.seq2seq_model import EncoderRNN, AttnDecoderRNN

eng = spacy.load("en_core_web_sm")  # Load the English model to tokenize English text
zh = spacy.load("zh_core_web_sm")
MAX_LENGTH = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
plt.switch_backend('agg')


class Seq2SeqModule(pl.LightningModule):
    def __init__(self, encoder, decoder, loss, plot_losses, learning_rate=2e-5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.lr = learning_rate
        self.plot_losses = plot_losses

    def forward(self, x):
        encoder_outputs, encoder_hidden = self.encoder(x)
        return self.decoder(encoder_outputs, encoder_hidden)

    def compute_loss(self, y_hat, y):
        return self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        encoder_outputs, encoder_hidden = self.encoder(x)
        x_hat, decoder_hidden, attentions = self.decoder(encoder_outputs, encoder_hidden, y)
        loss = self.compute_loss(x_hat, y)
        self.plot_losses.append(loss)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        encoder_outputs, encoder_hidden = self.encoder(x)
        x_hat, _, _ = self.decoder(encoder_outputs, encoder_hidden, y)
        test_loss = self.compute_loss(x_hat, y)

        self.log('test_loss', test_loss, on_epoch=True, on_step=False, prog_bar=True)
        return test_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        encoder_outputs, encoder_hidden = self.encoder(x)
        x_hat, _, _ = self.decoder(encoder_outputs, encoder_hidden, y)
        val_loss = self.loss(x_hat, y)

        self.log('val_loss', val_loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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


# 删除行多余数据
def removeAttribution(row):
    """
    Function to keep the first two elements in a tuple
    """
    return row[:2]


def engTokenize(text):
    """
    Tokenize an English text and return a list of tokens
    """

    return [token.text for token in eng.tokenizer(text)]


def zhTokenize(text):
    """
    Tokenize a German text and return a list of tokens
    """

    return [token.text for token in zh.tokenizer(text)]


def getTokens(data_iter, place):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """

    for english, chinese in data_iter:
        if place == 0:
            yield engTokenize(english)
        else:
            yield zhTokenize(chinese)


def getTransform(vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        T.Truncate(max_seq_len=MAX_LENGTH - 2),
        ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(1, begin=True),
        ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(2, begin=False),
        T.ToTensor(padding_value=0),
        T.PadTransform(MAX_LENGTH, 0)
    )
    return text_tranform


# def applyTokenize(sequence_pair):
#     """
#     Apply transforms to sequence of tokens in a sequence pair
#     """
#     return (engTokenize(sequence_pair[0]), zhTokenize(sequence_pair[1]))


# def applyTransform(sequence_pair):
#     """
#     Apply transforms to sequence of tokens in a sequence pair
#     """
#
#     return (getTransform(sequence_pair[0]), getTransform(sequence_pair[1]))


def build_datapipes(root_dir="."):
    data_pipe = dp.iter.IterableWrapper([root_dir])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
    data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t')
    data_pipe = data_pipe.map(removeAttribution)
    return data_pipe


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    FILE_PATH = 'data/cmn.txt'
    data_pipe = build_datapipes(FILE_PATH)
    source_vocab = build_vocab_from_iterator(
        getTokens(data_pipe, 0),
        min_freq=2,
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )
    source_vocab.set_default_index(source_vocab['<unk>'])

    target_vocab = build_vocab_from_iterator(
        getTokens(data_pipe, 1),
        min_freq=2,
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )
    target_vocab.set_default_index(target_vocab['<unk>'])
    src_data = []
    target_data = []
    for src, target in data_pipe:
        src_token, target_token = engTokenize(src), zhTokenize(target)
        en_data, zh_data = getTransform(source_vocab)(src_token), getTransform(target_vocab)(target_token)
        src_data.append(en_data)
        target_data.append(zh_data)

    # data_pipe = data_pipe.map(applyTokenize)
    # data_pipe = data_pipe.map(lambda x: (source_vocab(x[0]), target_vocab(x[1])))
    # data_pipe = data_pipe.map(applyTransform)
    dataset = TensorDataset(torch.stack(src_data), torch.stack(target_data))
    # 定义模型
    hidden_size = 128
    batch_size = 8
    input_size = len(source_vocab)
    output_size = len(target_vocab)
    total_size = len(dataset)
    print(f"input: {input_size}")
    print(f"output: {output_size}")
    print(f"dataloader len: {total_size}")
    # 损失函数
    loss = nn.CrossEntropyLoss(ignore_index=1)
    lr = 1e-4
    # lighting模型
    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_size, max_length=MAX_LENGTH).to(device)
    plot_losses = []
    module = Seq2SeqModule(encoder, decoder, loss, plot_losses, learning_rate=lr)
    # 回调函数
    early_stop_callback = EarlyStopping(monitor="train_loss",
                                        patience=5,
                                        mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(
        monitor='train_loss',
        save_top_k=1,
        mode='min',
        filename="seq2seq-net-{epoch:02d}-{train_loss:.2f}"
    )
    # 实例化trainer
    trainer = Trainer(max_epochs=100,
                      accelerator="gpu",
                      devices=1,
                      profiler="simple",
                      callbacks=[early_stop_callback, ckpt_callback, lr_monitor],
                      enable_progress_bar=True,
                      enable_model_summary=True)
    # 训练
    dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1)
    # for src, target in dl:
    #     print(f"Feature batch shape: {src.size()}")
    #     print(f"Labels batch shape: {target.size()}")

    trainer.fit(model=module, train_dataloaders=dl)
    print("Train seq2seq model done")
    showPlot(plot_losses)
