import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from net import network
from dataset import dataloader
from dataset import data_prefetcher
from utils.utils import plot_classes_preds


def train(args):
    model = network.get_classifier(args.model_name, num_classes=10).cuda()
    dataset = dataloader.TorchLoader(args.dataset_name, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    train_prefetcher = data_prefetcher.DataPrefetcher(dataset.train_loader)
    test_prefetcher = data_prefetcher.DataPrefetcher(dataset.test_loader)
    len_train_dataset = len(dataset.train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    writer = SummaryWriter('./runs/test')

    for epoch in range(args.epoch):
        running_loss = 0.
        for i in range(len(train_prefetcher)):
            st_time = time.time()
            inputs, labels = dataset.read_train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch * len_train_dataset + i)
            loss.backward()
            optimizer.step()

            end_time = time.time()
            writer.add_scalar('Train/time_pre_step', end_time - st_time, epoch * len_train_dataset + i)

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            del loss

            if i % 700 == 699:
                print('draw training predictions')
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(model, inputs, labels),
                                  global_step=epoch * len_train_dataset + i)

            if i % 700 == 699:
                with torch.no_grad():
                    TP = FP = TN = FN = 0
                    for j in range(len(test_prefetcher)):
                        images, labels = dataset.read_test()
                        output = model(images)
                        _, class_preds_batch = torch.max(output, 1)
                        np_labels = labels.cpu().numpy()
                        np_preds = class_preds_batch.cpu().numpy()

                        TP += sum(np_labels[np.where(np_preds == 1)[0]] == 1)
                        FP += sum(np_labels[np.where(np_preds == 1)[0]] == 0)
                        TN += sum(np_labels[np.where(np_preds == 0)[0]] == 0)
                        FN += sum(np_labels[np.where(np_preds == 0)[0]] == 1)

                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    writer.add_scalar("Val/precision", precision, epoch * len_train_dataset + i)
                    writer.add_scalar("Val/recall", recall, epoch * len_train_dataset + i)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--dataset_name', default='CIFAR10')
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    train(args)
