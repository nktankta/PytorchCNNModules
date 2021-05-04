import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from module_easyModel import EasyModel
from PytorchCNNModules import *

test_module = InceptionD
mode = "normal"
mode_args = {}
module_kwargs = {"activation":nn.ReLU,"activation_kwargs":{"inplace":True},"norm_layer":nn.BatchNorm2d}
epochs = 2

device = "cuda"

transform = transforms.Compose(
    [transforms.RandomRotation(15),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
criterion = nn.CrossEntropyLoss()


def test_train_model():
    print("start testing")
    net = EasyModel(1,10,test_module,mode=mode,mode_kwargs=mode_args,module_kwargs=module_kwargs).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct / total)))
    assert float(correct / total)>0.25


