import torch
import torch.nn as nn
import torch.optim as optim

from net import network
from dataset import dataloader


def train():
    model = network.get_network('resnet18', num_classes=10).cuda()
    dataset = dataloader.TorchLoader('CIFAR10', train_batch_size=64, test_batch_size=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.
        for i in range(len(dataset.train_loader)):
            inputs, labels = dataset.read_train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    train()
