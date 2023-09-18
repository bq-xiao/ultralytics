import torch

from data import vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, train_dataloader
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
n = 0
print(f"val_dataloader len: {len(list(train_dataloader))}")
for src, target in train_dataloader:
    input = src[:, 0]
    output = target[:, 0]
    print(f"src:{vocab_transform['eng'].lookup_tokens(list(input.cpu().numpy()))}")
    print(f"target:{vocab_transform['zh'].lookup_tokens(list(output.cpu().numpy()))}")
    # predict with the model
    predict = model(src)
    print(f"predict:{vocab_transform['zh'].lookup_tokens(list(predict.cpu().numpy()))} \n")
    n = n + 1
    if n > 20:
        break
