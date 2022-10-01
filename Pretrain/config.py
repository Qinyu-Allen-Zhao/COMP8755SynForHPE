import argparse
from yacs.config import CfgNode as CN

# Configuration variables
cfg = CN()
cfg.MODEL = CN()

cfg.MODEL.PRETRAINED = ""

cfg.MODEL.EXTRA = CN()
cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
cfg.MODEL.EXTRA.DECONV_WITH_BIAS = False
cfg.MODEL.EXTRA.NUM_DECONV_LAYERS = 3
cfg.MODEL.EXTRA.NUM_DECONV_FILTERS = {256, 256, 256}
cfg.MODEL.EXTRA.NUM_DECONV_KERNELS = {4, 4, 4}
cfg.MODEL.EXTRA.NUM_LAYERS = 50
