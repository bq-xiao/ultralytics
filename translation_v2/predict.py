import torch

from data import SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, text_transform
from model import Seq2SeqTransformer, Seq2SeqModule

torch.manual_seed(0)
vocab_dic = {}
print("loading vocab ...")
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_dic[ln] = torch.load(ln + ".pt")

SRC_VOCAB_SIZE = len(vocab_dic[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_dic[TGT_LANGUAGE])
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

print("loading model ...")
model = Seq2SeqModule.load_from_checkpoint(
    "lightning_logs/version_1/checkpoints/seq2seq-transformer-epoch=50-val_loss=0.05.ckpt",
    transformer=transformer, loss=loss_fn, lr=lr
)
# disable randomness, dropout, etc...
model.eval()
while True:
    src = input("<:")
    if src is None or '' == src:
        continue
    if src == 'q':
        print("good bye! shutting down ... ")
        exit(0)
    data = text_transform[SRC_LANGUAGE](src).view(-1, 1)
    predict = model(data)
    print(f"predict:{vocab_dic[TGT_LANGUAGE].lookup_tokens(list(predict.cpu().numpy()))} \n")
