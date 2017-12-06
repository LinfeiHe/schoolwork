import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import pretrainedmodels
from PIL import Image


class TestData(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir)
                                if any(x.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx])
        print(self.image_filenames[idx], idx)
        id = os.path.basename(self.image_filenames[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, id


def softmax(x):
    a = np.exp(x)
    return a / np.array([np.sum(a, 1)]).T



def get_outputs(image_dir, filename):
    models_name = ['resnet152', 'vgg19_bn', 'densenet161', 'nasnetalarge']
    res = {}
    res_labels = {}
    for name in models_name:
        if name == 'densenet161':
            model_ft = torch.load('model_pretrained_densenet161.pkl')
        elif name == 'resnet152':
            model_ft = torch.load('model_pretrained_resnet152.pkl')
        elif name == 'vgg19_bn':
            model_ft = torch.load('model_pretrained_vgg19.pkl')

        data_transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        if name == 'nasnetalarge':
            model_ft = pretrainedmodels.nasnetalarge(num_classes=1000, pretrained='imagenet')
            data_transforms = transforms.Compose([
                transforms.Scale(377),
                transforms.CenterCrop(331),
                transforms.ToTensor(),
                transforms.Normalize(mean=model_ft.mean,
                                     std=model_ft.std)])
            model_ft = torch.load('model_pretrained_nasnet.pkl')
        use_gpu = torch.cuda.is_available()
        model_ft.eval()


        test_dataset = TestData(image_dir, data_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        since = time.time()
        temp = []
        temp_list = []
        for i, batch in enumerate(test_dataloader):
            inputs, cid = batch
            temp_list.append(cid)
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            outputs = model_ft(inputs)
            temp.append(softmax(outputs.data.cpu().numpy()))
            if i % 200 == 199:
                print('iter:{}'.format(i+1))
        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        res[name] = np.concatenate(temp)
        res_labels[name] = [y for x in temp_list for y in x]
        print('{} finish'.format(name))

    torch.save(res, filename)
    torch.save(res_labels, filename + '_label')
    return res


image_dir = '/home/helinfei/PycharmProjects/ai_challenge/scene/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
val_outputs = get_outputs(image_dir, 'val_outputs.pkl')
image_dir = './data/test/'
get_outputs(image_dir, 'test_outputs.pkl')

