import torch
import torchvision
import param 
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

random_seed = 1
torch.backends.cudnn.enabled = False
def getDataset():

  train_set = torchvision.datasets.MNIST('./data/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ]))

  test_set =torchvision.datasets.MNIST('./data/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ]))
  num_classes =10
  inputs =1

  return train_set, test_set, inputs, num_classes

def getDataloader(trainset, testset, valid_size, batch_size_train,batch_size_test, num_workers):

  num_train = len(trainset)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(valid_size*num_train))

  train_idx, valid_idx = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
    sampler=train_sampler, num_workers=num_workers)
  valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, 
    sampler=valid_sampler, num_workers=num_workers)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, 
    num_workers=num_workers)

  
  return train_loader, valid_loader, test_loader




