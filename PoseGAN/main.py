from __future__ import print_function

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from config import _C as cfg
from discriminator import Discriminator
from generator import Generator
from utils import weights_init

# Set random seed for reproducibility
manualSeed = 2022
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
syn_dataset = dset.ImageFolder(root=cfg.data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(cfg.image_size),
                                   transforms.CenterCrop(cfg.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
real_dataset = dset.ImageFolder(root=cfg.real_root,
                                transform=transforms.Compose([
                                    transforms.Resize(cfg.image_size),
                                    transforms.CenterCrop(cfg.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

# Create the dataloader
syn_dataloader = torch.utils.data.DataLoader(syn_dataset, batch_size=cfg.batch_size,
                                             shuffle=True, num_workers=cfg.workers)
real_dataloader = torch.utils.data.DataLoader(syn_dataset, batch_size=cfg.batch_size,
                                              shuffle=True, num_workers=cfg.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and cfg.ngpu > 0) else "cpu")

# Plot some training images
syn_batch = next(iter(syn_dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(syn_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Create the generator
netG = Generator(cfg.ngpu, cfg.nc, cfg.ngf).to(device)
# Create the Discriminator
netD = Discriminator(cfg.ngpu, cfg.nc, cfg.ndf).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (cfg.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(cfg.ngpu)))
    netD = nn.DataParallel(netD, list(range(cfg.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)
netD.apply(weights_init)

# Print the model
print(netG)
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# Grab a batch of real images from the dataloader
syn_batch = next(iter(syn_dataloader))
real_batch = next(iter(real_dataloader))
#  the progression of the generator
fixed_images = syn_batch[0]

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(cfg.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(syn_dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_images = real_batch[i % len(real_batch)].to(device)
        b_size = real_images.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_images).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate fake image batch with G
        syn_images = data[0].to(device)
        fake = netG(syn_images)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, cfg.num_epochs, i, len(syn_dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    # Check how the generator is doing by saving G's output
    with torch.no_grad():
        fake = netG(fixed_images).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    torch.save(netD.state_dict(), os.path.join(cfg.save_path, "%d_discriminator.pt" % epoch))
    torch.save(netG.state_dict(), os.path.join(cfg.save_path, "%d_generator.pt" % epoch))

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/iteration_loss.jpg", dpi=300)
plt.show()

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(syn_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
plt.savefig("results/real_images.jpg", dpi=300)
plt.show()

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.savefig("results/fake_images.jpg", dpi=300)
plt.show()

# fig = plt.figure(figsize=(8,8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#
# HTML(ani.to_jshtml())
