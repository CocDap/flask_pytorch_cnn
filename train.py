from model import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import param
import data
import torch
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)



def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % param.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


if __name__ =="__main__":
  train_set, test_set, inputs, num_classes = data.getDataset()
  train_loader, valid_loader, test_loader = data.getDataloader(train_set, test_set,param.valid_size, param.batch_size_train,param.batch_size_test, param.num_workers )
  network = Net()

  optimizer = optim.SGD(network.parameters(), lr=param.learning_rate,
                  momentum=param.momentum)
  test()
  for epoch in range(1, param.n_epochs + 1):
    train(epoch)
    test()








