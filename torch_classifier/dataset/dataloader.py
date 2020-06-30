import torch
import torchvision.datasets
import torchvision.transforms as transforms
from tfrecord.torch.dataset import TFRecordDataset


class TFRecordLoader(object):
    def __init__(self, tfrecord_path, batch_size):
        description = {"image": "byte", "label": "float"}
        index_path = None
        dataset = TFRecordDataset(tfrecord_path, index_path, description)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def read(self):
        return next(iter(self.loader))


class TorchLoader(object):
    def __init__(self, dataset_name,
                 train_batch_size=16,
                 test_batch_size=8,
                 data_path=None):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if hasattr(torchvision.datasets, dataset_name):
            train_set = getattr(torchvision.datasets, dataset_name)(root='./data',
                                                                    train=True,
                                                                    download=True,
                                                                    transform=transform)
            self.train_loader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=train_batch_size,
                                                            shuffle=True,
                                                            num_workers=1)
            test_set = getattr(torchvision.datasets, dataset_name)(root='./data',
                                                                   train=False,
                                                                   download=True,
                                                                   transform=transform)
            self.test_loader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=test_batch_size,
                                                           shuffle=True,
                                                           num_workers=1)

        else:
            raise ImportError('unknown dataset {}, only torchvision seems not to support'.format(dataset_name))

    def read_train(self):
        images, labels = next(iter(self.train_loader))
        return images.cuda(), labels.cuda()

    def read_test(self):
        images, labels = next(iter(self.test_loader))
        return images.cuda(), labels.cuda()
