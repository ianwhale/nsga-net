# dataloader.py

import torch
import datasets
import torch.utils.data
import torchvision.transforms as transforms

class Dataloader:
    def __init__(self, args):
        self.args = args

        self.loader_input = args.loader_input
        self.loader_label = args.loader_label

        self.split_test = args.split_test
        self.split_train = args.split_train
        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train
        self.resolution = (args.resolution_wide, args.resolution_high)

        self.input_filename_test = args.input_filename_test
        self.label_filename_test = args.label_filename_test
        self.input_filename_train = args.input_filename_train
        self.label_filename_train = args.label_filename_train

        if self.dataset_train_name == 'LSUN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                db_path=args.dataroot,
                classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'CIFAR10' or self.dataset_train_name == 'CIFAR100':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(self.resolution, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )

        elif self.dataset_train_name == 'MYCIFAR10' or self.dataset_train_name == 'MYCIFAR100':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(self.resolution, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )

        elif self.dataset_train_name == 'CocoCaption' or self.dataset_train_name == 'CocoDetection':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'STL10' or self.dataset_train_name == 'SVHN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataroot, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'MNIST':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(
                root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )

        elif self.dataset_train_name == 'ImageNet':
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.dataset_train = datasets.ImageFolder(
                root=self.args.dataroot + self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            )

        elif self.dataset_train_name == 'FRGC':
            self.dataset_train = datasets.ImageFolder(
                root=self.args.dataroot + self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'Folder':
            self.dataset_train = datasets.ImageFolder(
                root=self.args.dataroot + self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_train_name == 'FileListLoader':
            self.dataset_train = datasets.FileListLoader(
                self.input_filename_train, self.label_filename_train,
                self.split_train, self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                transform_test=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
            )

        elif self.dataset_train_name == 'FolderListLoader':
            self.dataset_train = datasets.FileListLoader(
                self.input_filename_train, self.label_filename_train,
                self.split_train, self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                transform_test=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
            )

        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == 'LSUN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                db_path=args.dataroot, classes=['bedroom_val'],
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'CIFAR10' or self.dataset_test_name == 'CIFAR100':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )

        elif self.dataset_test_name == 'MYCIFAR10' or self.dataset_test_name == 'MYCIFAR100':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            )

        elif self.dataset_test_name == 'CocoCaption' or self.dataset_test_name == 'CocoDetection':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'STL10' or self.dataset_test_name == 'SVHN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataroot, split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'MNIST':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(
                root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )

        elif self.dataset_test_name == 'ImageNet':
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.dataset_test = datasets.ImageFolder(
                root=self.args.dataroot + self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            )

        elif self.dataset_test_name == 'FRGC':
            self.dataset_test = datasets.ImageFolder(
                root=self.args.dataroot + self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'Folder':
            self.dataset_test = datasets.ImageFolder(
                root=self.args.dataroot + self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            )

        elif self.dataset_test_name == 'FileListLoader':
            self.dataset_test = datasets.FileListLoader(
                self.input_filename_test, self.label_filename_test,
                self.split_train, self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
            )

        elif self.dataset_test_name == 'FolderListLoader':
            self.dataset_test = datasets.FileListLoader(
                self.input_filename_test, self.label_filename_test,
                self.split_train, self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Resize(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
            )
        else:
            raise(Exception("Unknown Dataset"))

    def create(self, flag=None):
        dataloader = {}
        if flag == "Train":
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )
            return dataloader

        if flag == "Test":
            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True
            )
            return dataloader

        if flag is None:
            dataloader['train'] = torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=True, pin_memory=True
            )

            dataloader['test'] = torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.args.batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False, pin_memory=True)
            return dataloader
