from datasets import KMNIST, TIMG
import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset



def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def uniform_noise(labels, num_class, noise_level):
    noise_level = noise_level# * num_class / (num_class-1) 
    rand_class = np.random.randint(num_class,size=labels.shape[0])
    checks = np.random.rand(labels.shape[0])
    noise_tag = np.zeros_like(labels)
    for i in range(labels.shape[0]):
      if checks[i] < noise_level:
        labels[i] = rand_class[i]
        noise_tag[i] = 1
    return labels, noise_tag


def classDep_noise(labels, num_class, noise_level): 
    checks = np.random.rand(labels.shape[0])
    noise_tag = np.zeros_like(labels)
    for i in range(labels.shape[0]):
      if checks[i] < noise_level:
        labels[i] = (labels[i] + 1) % num_class
        noise_tag[i] = 1
    return labels, noise_tag


def classDep_news20(labels, noise_level):
    transY = np.arange(20)
    checks = np.random.rand(labels.shape[0])
    transY[2] = 5
    transY[5] = 2
    transY[3] = 4
    transY[4] = 3
    transY[7] = 8
    transY[8] = 7
    transY[9] = 10
    transY[10] = 9
    transY[11] = 12
    transY[12] = 11
    transY[15] = 18
    transY[18] = 15
    noise_tag = np.zeros_like(labels)
    for i in range(labels.shape[0]):
      if checks[i] < noise_level:
        labels[i] = transY[labels[i]]
        noise_tag[i] = 1
    return labels, noise_tag


def classDep_letter(labels, noise_level):
    transY = np.arange(26)
    checks = np.random.rand(labels.shape[0])
    transY[1] = 3
    transY[3] = 1
    transY[2] = 6
    transY[6] = 2
    transY[4] = 5
    transY[5] = 4
    transY[7] = 13
    transY[13] = 7
    transY[8] = 11
    transY[11] = 8
    transY[10] = 23
    transY[23] = 10
    transY[12] = 22
    transY[22] = 12
    transY[14] = 16
    transY[16] = 14
    transY[15] = 17
    transY[17] = 15
    transY[20] = 21
    transY[21] = 20
    noise_tag = np.zeros_like(labels)
    for i in range(labels.shape[0]):
      if checks[i] < noise_level:
        labels[i] = transY[labels[i]]
        noise_tag[i] = 1
    return labels, noise_tag


def classDep_vowel(labels, noise_level):
    transY = np.arange(11)
    checks = np.random.rand(labels.shape[0])
    transY[0] = 1
    transY[1] = 0
    transY[2] = 3
    transY[3] = 2
    transY[4] = 5
    transY[5] = 4
    transY[6] = 7
    transY[7] = 6
    transY[8] = 9
    transY[9] = 8
    noise_tag = np.zeros_like(labels)
    for i in range(labels.shape[0]):
      if checks[i] < noise_level:
        labels[i] = transY[labels[i]]
        noise_tag[i] = 1
    return labels, noise_tag

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    #print(output)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size).detach().numpy())
    return res


def balanced_accuracy(output, target, num_class, topk=(1,)):
    accs = []
    for cls in range(num_class):
      mask = (target == cls)
      tmp_acc = accuracy(output[mask],target[mask],topk)
      accs.append(tmp_acc)
    return np.mean(accs, axis=0)


class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets[0]
       self.noise_tag = targets[1]
       self.mode = mode
       self.transform_train = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                             ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        noise_tag = self.noise_tag[idx]
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target, noise_tag, int(idx)

    def get_labels(self):
        return np.array(self.targets).reshape(-1)


class TabularDataset(Dataset):
    def __init__(self, data, targets):
       self.data = data
       self.targets = targets[0]
       self.noise_tag = targets[1]
       
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx] 
        noise_tag = self.noise_tag[idx]
        return data, target, noise_tag, int(idx)

    def get_labels(self):
        return np.array(self.targets).reshape(-1)



class ImageDataset_cifar(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets[0]
       self.noise_tag = targets[1]
       self.mode = mode
       self.transform_train = transforms.Compose([                                                
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        noise_tag = self.noise_tag[idx]
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target, noise_tag, int(idx)

    def get_labels(self):
        return np.array(self.targets).reshape(-1)

