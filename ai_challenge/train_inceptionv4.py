import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import pretrainedmodels


model_ft = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_ft.mean,
                             std=model_ft.std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(341),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_ft.mean,
                             std=model_ft.std)
    ]),
}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=28, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_path = 0.0
            running_corrects1 = 0
            running_corrects3 = 0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                _, preds = torch.topk(outputs.data, 3, 1)
                loss = criterion(outputs, labels)
                pred = preds.t()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_loss_path += loss.data[0]
                correct = pred.eq(labels.data.view(1, -1).expand_as(pred))
                running_corrects1 += torch.sum(correct[:1].view(-1).float())
                running_corrects3 += torch.sum(correct[:3].view(-1).float())

                if i % 200 == 199:
                    print('{} iter: {:.4f} Loss_path: {:.4f}'.format(phase, i+1, running_loss_path))
                    running_loss_path = 0.0

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc1 = running_corrects1 / dataset_sizes[phase]
            epoch_acc3 = running_corrects3 / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc1: {:.4f} Acc3: {:.4f}'.format(
                phase, epoch_loss, epoch_acc1, epoch_acc3))

            # deep copy the model
            if phase == 'val' and epoch_acc3 > best_acc:
                best_acc = epoch_acc3
                best_model_wts = model.state_dict()


        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'best_model_wts_inceptionv4.pkl')
    torch.save(best_acc, 'best_acc_inceptionv4.pkl')
    return model


def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

num_ftrs = model_ft.classif.in_features
model_ft.classif = nn.Linear(num_ftrs, 80)

if use_gpu:
    model_ft = nn.DataParallel(model_ft).cuda()


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)

visualize_model(model_ft)
plt.pause(0)

"""
train Loss: 0.0551 Acc1: 0.5876 Acc3: 0.7756
val iter: 200.0000 Loss_path: 206.1954
val Loss: 0.0376 Acc1: 0.7249 Acc3: 0.9003

Training complete in 76m 27s
Best val Acc: 0.900281
"""