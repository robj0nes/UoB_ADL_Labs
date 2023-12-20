from transformers import AutoTokenizer, BertForSequenceClassification
import timeit
import torch

b_cpu = torch.rand((10000, 10000), device='cpu')
b_mps = torch.rand((10000, 10000), device='mps')

print('cpu', timeit.timeit(lambda: b_cpu @ b_cpu, number=100))
print('mps', timeit.timeit(lambda: b_mps @ b_mps, number=100))