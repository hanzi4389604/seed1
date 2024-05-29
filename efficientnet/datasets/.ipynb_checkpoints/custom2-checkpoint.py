import os
from glob import glob

from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import pil_loader


class CustomDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            image_dir = os.path.join(self.root, 'train')
        else:
            image_dir = os.path.join(self.root, 'valid')

        self.paths = glob(os.path.join(image_dir, '*.jpg'))

    def __getitem__(self, index):
        path = self.paths[index]
        img = pil_loader(path)
        target = 0 if 'cat' in path else 1

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.paths)


class CustomDataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, batch_size: int, train: bool = True, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = CustomDataset(root, train=train, transform=transform)
        super(CustomDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)
####
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
train_augs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224])])
test_augs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224])
])
######
def custom_dataloaders(root='/home/l/20211218 practice/', image_size=1080, batch_size=8, **kwargs):
#    train_loader = CustomDataLoader(root, image_size, batch_size=batch_size, train=True, **kwargs)
#    test_loader = CustomDataLoader(root, image_size, batch_size=batch_size, train=False, **kwargs)

    ######
    train_loader=DataLoader(ImageFolder(os.path.join(root,
            'hotdog1/train'),transform=train_augs), batch_size,shuffle=True)
    test_loader=DataLoader(ImageFolder(os.path.join(root,
            'hotdog1/test'),transform=test_augs), batch_size)
    #######
    
    return train_loader, test_loader
