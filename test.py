import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class MyDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        print("调用")
        print("item是:",item)
        print(self.idxs[item])
        image, label = self.dataset[self.idxs[item]]
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)


train_dataset1 = MyDataset(train_dataset, [1,2,3,5])
print(len(train_dataset1[0]))
print(train_dataset1[0])
print(train_dataset1[1])
print(train_dataset1[2])
print(train_dataset1[3])
