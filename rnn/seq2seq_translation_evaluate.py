import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator

from seq2seq_model import EncoderRNN, AttnDecoderRNN
from seq2seq_translation_trainer import build_datapipes, getTokens, engTokenize, getTransform, Seq2SeqModule

device = torch.device("cpu")
EOS_token = 1

if __name__ == '__main__':
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

    # data_pipe = data_pipe.map(applyTokenize)
    # data_pipe = data_pipe.map(lambda x: (source_vocab(x[0]), target_vocab(x[1])))
    # data_pipe = data_pipe.map(applyTransform)

    # 定义模型
    hidden_size = 128
    batch_size = 64
    input_size = 4811
    output_size = 7350
    MAX_LENGTH = 256
    # 损失函数
    loss = nn.CrossEntropyLoss(ignore_index=1)
    lr = 1e-4
    # lighting模型
    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_size, max_length=MAX_LENGTH).to(device)
    model = Seq2SeqModule.load_from_checkpoint("seq2seq-net.ckpt",
                                               encoder=encoder,
                                               decoder=decoder,
                                               loss=loss,
                                               plot_losses=[],
                                               learning_rate=lr)

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for src, target in data_pipe:
            print('>', src)
            print('=', target)
            src_token = engTokenize(src)
            en_data = getTransform(source_vocab)(src_token)

            en_data = en_data.unsqueeze(0)
            decoder_outputs, decoder_hidden, decoder_attn = model(en_data)
            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            decoded_ids = decoded_ids.view(-1)
            decoded_words = []
            print(f"output shape:{decoded_ids.shape}")
            for idx in decoded_ids:
                index = idx.item()
                if index == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(target_vocab.lookup_token(index))

            output_sentence = ' '.join(decoded_words)
            print('<', output_sentence)
