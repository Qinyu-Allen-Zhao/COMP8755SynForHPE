from yacs.config import CfgNode as CN

_C = CN()

# Root directory for dataset
_C.data_root = "/content/syn_images/"
_C.real_root = "/content/real_images/mpii/"
# Number of workers for dataloader
_C.workers = 2
# Batch size during training
_C.batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
_C.image_size = 64
# Number of channels in the training images. For color images this is 3
_C.nc = 3
# Size of z latent vector (i.e. size of generator input)
_C.nz = 100
# Size of feature maps in generator
_C.ngf = 64
# Size of feature maps in discriminator
_C.ndf = 64
# Number of training epochs
_C.num_epochs = 5
# Learning rate for optimizers
_C.lr = 0.0002
# Beta1 hyperparam for Adam optimizers
_C.beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
_C.ngpu = 1
# The path for saving the model checkpoints
_C.save_path = "checkpoints/"