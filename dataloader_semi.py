
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt
import random
from glob import glob
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import h5py
import SimpleITK as sitk
__all__ = ['GastricCancerDataset_Semi', 'Synapsedataset_Semi','RandomResize', 'Resize', 'CenterCrop', 'RandomCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'ToTensor', 'Normalize','RandomGenerator']


class GastricCancerDataset_Semi(Dataset):
    def __init__(self, image_path="dataset/GastricCancer/resize",split="train",supervised=True, percent_labeled=0.2,transform=None):
       self.image_path= image_path
       self.transform = transform
       if supervised:
            self.sample_list= open(os.path.join(self.image_path+"/list", split+"_labeled_"+str(percent_labeled)+'.txt')).readlines()
       else:
            self.sample_list = open(os.path.join(self.image_path+"/list", split+"_unlabeled_"+str(percent_labeled)+'.txt')).readlines()
       self.image_list=[]
       self.label_list=[]
       for sample in self.sample_list:
           self.image_list.append(self.image_path+"/train/image/"+sample.strip("\n"))
           self.label_list.append(self.image_path+"/train/label/"+sample.strip("\n"))
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        sample_path = self.image_list[index]
        sample = Image.open(sample_path)
        label_path = self.label_list[index]
        sample_name = self.image_list[index].split("/")[-1].split(".")[0]
        """
        print(self.image_list[index].split(".")[0]+"_1stHO.png")
        print(label_path)
        assert self.image_list[index].split(".")[0]+"_1stHO.png" == label_path
        """
        #print(sample_name)
        label = Image.open(label_path)
        #print("convert 前:", np.array(label).shape)
        label = label.convert("L")
        #print("convert 后:",np.array(label).shape)
        Sample = {"name": sample_name,
                  "image": sample,
                  "label": label}
        if self.transform:
            Sample = self.transform(Sample)

        return Sample   


class Synapsedataset_Semi(Dataset):
    def __init__(self, base_dir, list_dir, split,supervised=True, num_case_labled=2,transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        if supervised:
            
            self.sample_list= open(os.path.join(list_dir, self.split+"_labeled_"+str(num_case_labled)+'.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split+"_unlabeled_"+str(num_case_labled)+'.txt')).readlines()
        
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir+"/train_npz", slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir+"/test_vol_h5" + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        
        sample['name'] = self.sample_list[idx].strip('\n')
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomResize(object):

    def __call__(self, sample):
        name, image, label = sample["name"], sample["image"], sample["label"]
        h, w = image.size[:2]
        image_interpolation = Image.BILINEAR
        label_interpolation = Image.NEAREST

        a = 0.5 + random.random()*1.5

        output_size = (int(h*a), int(w*a))

        image = image.resize(output_size, image_interpolation)
        label = label.resize(output_size, label_interpolation)

        return {"name": name,
                "image": image,
                "label": label}


class Resize(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        name, image, label = sample["name"], sample["image"], sample["label"]
        h, w = image.size[:2]
        image_interpolation = Image.BILINEAR
        label_interpolation = Image.NEAREST
        image = image.resize(self.output_size,image_interpolation)
        label = label.resize(self.output_size,label_interpolation)

        return {"name": name,
                "image": image,
                "label": label}


class CenterCrop(object):

    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        name, image, label = sample["name"], sample["image"], sample["label"]
        image_h, image_w = image.size[:2]
        crop_h, crop_w = self.output_size
        crop_top = int(round((image_h-crop_h)/2.))
        crop_left = int(round((image_w-crop_w)/2.))

        image = image.crop((crop_left, crop_top, crop_left+crop_w, crop_top+crop_h))
        label = label.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))

        return {"name": name,
                "image": image,
                "label": label}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        name, image, label = sample["name"], sample["image"], sample["label"]
        h, w = image.size[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h-new_h)
        if w - new_w == 0:
            left = 0
        else:
            left = np.random.randint(0, w-new_w)
        image.crop((left, top, left+new_w, top+new_h))

        return {"name": name,
                "image": image,
                "label": label}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        name, image, label = sample["name"], sample["image"], sample["label"]

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return {"name": name,
                "image": image,
                "label": label}


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        name, image, label = sample['name'], sample['image'], sample['label']

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return {'name': name,
                'image': image,
                'label': label}


class ToTensor(object):
    def __call__(self, sample):
        name, image, label = sample['name'], sample['image'], sample['label']
        image = image.convert("L")
        image = np.array(image)
        label = np.array(label)
        image = image[:, :, None]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float().div(255)
        label = torch.from_numpy(label).float()
        return {'name': name,
                'image': image,
                'label': label}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        name, image, label = sample['name'], sample['image'], sample['label']
        image = image.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=image.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=image.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return {'name': name,
                'image': image,
                'label': label}

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label,name = sample['image'], sample['label'],sample['name']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long(),'name':name}
        return sample
if __name__ == "__main__":
    
    #dataset = Synapsedataset(base_dir="../dataset/Synapse",list_dir="../dataset/Synapse/lists_Synapse",split="train")
    dataset=SegTHORdataset(types="test")
    print(len(dataset))
    print(dataset[0]["image"].shape)
    print(dataset[0]["label"].shape)
    