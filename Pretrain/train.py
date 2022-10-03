import time
from tqdm import tqdm
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
img_path = '/content/drive/MyDrive/datasets/synthesis_dataset/'
ann_path = '/content/drive/MyDrive/datasets/syn_poses/'

# Transform
train_transform = transforms.Compose([
    # Normalize to [-1, 1]
    # because ToTensor will convert an image to [0, 1],
    # we use (i-0.5) / 0.5 to normalize it to [-1, 1]
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((256, 256)),
    # Data augmentation
    transforms.RandomHorizontalFlip(0.5),
    transforms.Pad(padding=4, fill=0),
    transforms.RandomCrop(size=(256, 256))
])

# Dataset instances
train_set = ImageDataset(img_path, ann_path,
                         transform=train_transform, target_transform=None)
# Data loader
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size, shuffle=True)

# Model, loss, optimizer, and writer for Tensorboard
gpu_available = torch.cuda.is_available()
model = get_pose_net(cfg, is_train=True)
print(model)

loss_func = nn.MSELoss()
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

for epoch in tqdm(range(num_epochs)):
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

    PATH = './checkpoints/pretrain_{:02d}.pth'.format(epoch)
    torch.save(model.state_dict(), PATH)

    train_acc, train_loss = eva_acc_and_loss(train_loader, model, gpu_available)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)

    # Print the model performance
    curr = time.time()
    print('Epoch: {}, Time: {}'.format(epoch, round(curr - end, 3)))
    print('---- train accuracy: {:.4}%, train loss: {:.4}'.format(train_acc, train_loss))
    end = curr
