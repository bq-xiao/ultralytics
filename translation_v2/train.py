import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from data import PAD_IDX, vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, train_dataloader, val_dataloader
from model import Seq2SeqModule, Seq2SeqTransformer

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

module = Seq2SeqModule(transformer, loss_fn, learning_rate=lr)
# 回调函数
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    patience=3,
                                    mode="min")
lr_monitor = LearningRateMonitor(logging_interval="step")
ckpt_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min',
    filename="seq2seq-transformer-{epoch:02d}-{val_loss:.2f}"
)
# 实例化trainer
trainer = Trainer(max_epochs=100,
                  accelerator="gpu",
                  devices=1,
                  profiler="simple",
                  callbacks=[early_stop_callback, ckpt_callback, lr_monitor],
                  log_every_n_steps=50,
                  enable_progress_bar=True,
                  enable_model_summary=True)
# 训练
# for src, target in val_dataloader:
#     print(f"Feature batch shape: {src.size()}")
#     print(f"Labels batch shape: {target.size()}")

trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
            ckpt_path='lightning_logs/version_0/checkpoints/seq2seq-transformer-epoch=42-val_loss=0.07.ckpt')
print("Train seq2seq model done")
