import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
def train(epoch):
  network.train()
  train_counter = []
  train_losses = []
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #  epoch, batch_idx * len(data), len(train_loader.dataset),
      #  100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      
def test_():
  network.eval()
  test_loss = 0
  correct = 0
  pred = []
  with torch.no_grad():
    for data, target in test_loader:
      #print(data.shape)
      #print(np.shape(target))
      #print(data)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred.append(output.data.max(1, keepdim=True)[1])
  return pred
  '''
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  '''

n_epochs = 8
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 1000

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

from mnist import MNIST
import numpy as np
mndata = MNIST(sys.argv[1])
images_array, labels_array = mndata.load_training()
images_array = np.asarray(images_array)
labels_array = np.asarray(labels_array)
#print(np.shape(images_array))

BATCH_SIZE = 32
X_train = images_array.reshape(images_array.shape[0], 1, 28, 28)

y_train = labels_array.reshape(labels_array .shape[0])

input_shape = (28, 28, 1)

X_train = X_train.astype('float32')

X_train /= 255

torch_X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) 

training = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

train_loader = torch.utils.data.DataLoader(training, batch_size = BATCH_SIZE, shuffle = False)

for epoch in range(1, n_epochs + 1):
  train(epoch)
  
X_test, y_test = mndata.load_testing()
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_test = X_test.astype('float32')
X_test /= 255
y_test = y_test.reshape(y_test.shape[0])
torch_X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False)

y_pred = test_()
y_pred = np.array(y_pred)
y_pred = y_pred.flatten()
#y_pred = [i.tolist() for i in y_pred]
#y_pred = [item for sublist in y_pred for item in sublist]
print(*y_pred, sep='\n')
#test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
#test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)