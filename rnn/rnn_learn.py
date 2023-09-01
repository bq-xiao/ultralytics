import spacy
import torch
from matplotlib import pyplot as plt
from torchtext.datasets import Multi30k

T = 1000  # 总共产⽣1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
plt.plot(time, x)
plt.xlim(1, 1000)
plt.xlabel('time')
plt.ylabel('x')
plt.show()

eng = spacy.load("en_core_web_sm")
