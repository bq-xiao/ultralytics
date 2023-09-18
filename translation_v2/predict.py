import torch

from data import vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, text_transform
from model import Seq2SeqTransformer, Seq2SeqModule

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
print(f"SRC_VOCAB_SIZE:{SRC_VOCAB_SIZE}")
print(f"TGT_VOCAB_SIZE:{TGT_VOCAB_SIZE}")
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
lr = 0.0001

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

model = Seq2SeqModule.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/seq2seq-transformer-epoch=42-val_loss=0.07.ckpt",
    transformer=transformer, loss=loss_fn, lr=lr
)
# disable randomness, dropout, etc...
model.eval()
while True:
    src = input("<:")
    if src is None or '' == src:
        continue
    if src == 'q':
        print("good bye!")
        break
    data = text_transform['eng'](src).view(-1, 1)
    predict = model(data)
    print(f"predict:{vocab_transform['zh'].lookup_tokens(list(predict.cpu().numpy()))} \n")
