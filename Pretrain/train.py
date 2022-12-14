import time
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from torch import optim
from torchvision import transforms
import glob

from config import cfg
from dataset import ImageDataset
from pose_resnet import get_pose_net


# Hyper-parameters
batch_size = 64
num_epochs = 100

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

print("Start training ...")
print("GPU avaiable: {}".format(gpu_available))
end = time.time()

checkpoints = glob.glob("./checkpoints/*.pth")
cp_id = max([int(cp[cp.find('_') + 1: -4]) for cp in checkpoints])
start = cp_id + 1
print('Load model pretrain_{:02d}.pth'.format(cp_id))
model.load_state_dict(torch.load('./checkpoints/pretrain_{:02d}.pth'.format(cp_id)))

for epoch in tqdm(range(start, num_epochs), ascii=True, desc="Epochs"):
    model.train()
    for i, data in tqdm(enumerate(train_loader), ascii=True, desc="Iterations"):
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

