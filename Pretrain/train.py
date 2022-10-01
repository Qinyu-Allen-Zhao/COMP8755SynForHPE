import time
import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from config import cfg
from dataset import ImageDataset
from evaluation import eva_acc_and_loss
from pose_resnet import get_pose_net


# Hyper-parameters
batch_size = 64
num_epochs = 140

# Dataset
train_img_file = './data/kmnist-npz/kmnist-train-imgs.npz'
train_ann_file = './data/kmnist-npz/kmnist-train-labels.npz'
val_img_file = './data/kmnist-npz/kmnist-val-imgs.npz'
val_ann_file = './data/kmnist-npz/kmnist-val-labels.npz'
test_img_file = './data/kmnist-npz/kmnist-test-imgs.npz'
test_ann_file = './data/kmnist-npz/kmnist-test-labels.npz'
# Transform
train_transform = transforms.Compose([
    # Normalize to [-1, 1]
    # because ToTensor will convert an image to [0, 1],
    # we use (i-0.5) / 0.5 to normalize it to [-1, 1]
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5,), (0.5,)),
    # Data augmentation
    transforms.RandomHorizontalFlip(0.5),
    transforms.Pad(padding=4, fill=0),
    transforms.RandomCrop(size=(28, 28))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5,), (0.5,))
])
# Dataset instances
train_set = ImageDataset(train_img_file, train_ann_file,
                         transform=train_transform, target_transform=None)
val_set = ImageDataset(val_img_file, val_ann_file,
                       transform=val_transform, target_transform=None)
test_set = ImageDataset(test_img_file, test_ann_file,
                        transform=val_transform, target_transform=None)
# Data loader
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size, shuffle=True)

# Model, loss, optimizer, and writer for Tensorboard
gpu_available = torch.cuda.is_available()
model = get_pose_net(cfg, is_train=True)
loss_func = nn.CrossEntropyLoss()
if gpu_available:
    model.cuda()
    loss_func.cuda()
optimizer = optim.Adam(model.parameters(),
                       lr=1e-3,
                       betas=(0.9, 0.999))
writer = SummaryWriter()

print("Start training ...")
print("GPU avaiable: {}".format(gpu_available))
end = time.time()
best_model_epoch = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for i, data in enumerate(train_loader):
        images, labels = data
        if gpu_available:
            images = images.cuda()
            labels = labels.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        out = model(images)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()

    PATH = './checkpoints/kmnist_net_{:02d}.pth'.format(epoch)
    torch.save(model.state_dict(), PATH)

    train_acc, train_loss = eva_acc_and_loss(train_loader, model, gpu_available)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)

    val_acc, val_loss = eva_acc_and_loss(val_loader, model, gpu_available)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

    if val_loss < best_loss:
        # Record the best model based on the val performance
        best_model_epoch = epoch
        best_loss = val_loss
        torch.save(model.state_dict(), './best_model.pth')

    # Print the model performance
    curr = time.time()
    print('Epoch: {}, Time: {}'.format(epoch, round(curr - end, 3)))
    print('---- train accuracy: {:.4}%, train loss: {:.4}'.format(train_acc, train_loss))
    print('---- validation accuracy: {:.4}%, validation loss: {:.4}'.format(val_acc, val_loss))
    end = curr

# Load and evaluate the best model
PATH = './best_model.pth'
best_model = get_pose_net(cfg, is_train=False, model_path=PATH)
if gpu_available:
    best_model.cuda()
test_acc, test_loss = eva_acc_and_loss(test_loader, best_model, gpu_available)
print()
print('Best Model')
print('---- epoch: {}'.format(best_model_epoch))
print('---- test accuracy: {:.4}%, test loss: {:.4}'.format(test_acc, test_loss))