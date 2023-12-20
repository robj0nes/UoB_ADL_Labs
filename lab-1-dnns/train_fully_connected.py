import torch
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     print("MPS selected")
#     device = torch.device("mps")

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x


def accuracy(probs: torch.FloatTensor, targets: torch.LongTensor) -> float:
    """
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    """
    preds = probs.argmax(1)
    return np.around((preds == targets).float().sum().item() / len(targets), decimals=4)


def setup_data():
    iris = datasets.load_iris()  # datasets are stored in a dictionary containing an array of features and targets
    iris.keys()
    preprocessed_features = (iris['data'] - iris['data'].mean(axis=0)) / iris['data'].std(axis=0)
    labels = iris['target']
    # train_test_split takes care of the shuffling and splitting process
    train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_features, labels,
                                                                                test_size=1 / 3)
    features = {
        'train': torch.tensor(train_features, dtype=torch.float32).to(device),
        'test': torch.tensor(test_features, dtype=torch.float32).to(device),
    }
    labels = {
        'train': torch.tensor(train_labels, dtype=torch.long).to(device),
        'test': torch.tensor(test_labels, dtype=torch.long).to(device),
    }
    return labels, features


labels, features = setup_data()

feature_count = 4
hidden_layer_size = 100
class_count = 3

model = MLP(feature_count, hidden_layer_size, class_count)
model.to(device)
optimiser = optim.SGD(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

import datetime
start = datetime.datetime.now()

for epoch in range(0, 1000):
    logits = model.forward(features['train'])
    loss = criterion(logits, labels['train'])

    print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
        epoch,
        accuracy(logits, labels['train']) * 100,
        loss.item()
    ))

    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

print("training time: {}".format(datetime.datetime.now() - start))

logits = model.forward(features['test'])
test_accuracy = accuracy(logits, labels['test']) * 100
print("test accuracy: {:2.2f}".format(test_accuracy))
