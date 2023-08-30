import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from models.discriminator.discriminator import Discriminator
from models.generator.generator import Generator
from models.generator.vgg16 import VGG16
from options.train_options import TrainOptions
from datasets.dataset import create_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer import train

def load_model(path, generator, discriminator, both=True):
    model_dict = torch.load(path)
    # generator
    G_dict = generator.state_dict()
    G_pre_dict = model_dict['generator']
    G_pred_dict = {k: v for k, v in G_pre_dict.items() if k in G_dict}
    G_dict.update(G_pred_dict)
    generator.load_state_dict(G_dict, strict=False)
    print(f"load pretrained G weights")
    if both:
        # discriminator
        D_dict = discriminator.state_dict()
        D_pre_dict = model_dict['discriminator']
        D_pred_dict = {k: v for k, v in D_pre_dict.items() if k in D_dict}
        D_dict.update(D_pred_dict)
        discriminator.load_state_dict(D_dict, strict=False)
        print(f"load pretrained D weights")

opts = TrainOptions().parse

device = torch.device(opts.device)
os.makedirs('{:s}'.format(opts.save_dir), exist_ok=True)

is_cuda = torch.cuda.is_available()
if is_cuda:
    
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('GPU number: ', n_gpu)
    opts.distributed = n_gpu > 1
    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

# model & load model
generator = Generator(opts)
discriminator = Discriminator(opts)
extractor = VGG16()

# cuda
if is_cuda:
    generator, discriminator, extractor = generator.to(device), discriminator.to(device), extractor.to(device)

# optimizer
if opts.finetune == True:
    print('Fine tune...')
    lr = opts.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = opts.gen_lr

generator_optim = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr * opts.D2G_lr)

# load checkpoints
if opts.pre_trained != '':
    ckpt_dict = torch.load(opts.pre_trained, map_location=lambda storage, loc: storage)
    opts.start_iter = ckpt_dict['n_iter']
    generator.load_state_dict(ckpt_dict['generator'])
    discriminator.load_state_dict(ckpt_dict['discriminator'])

    print('Starting from iter', opts.start_iter)
else:
    print('Starting from iter', opts.start_iter)


if opts.distributed:

    generator = nn.parallel.DistributedDataParallel(
        generator, 
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator, 
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )

# dataset
image_dataset = create_image_dataset(opts)
print(image_dataset.__len__())

image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    sampler=data_sampler(
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True
)

# training
train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda)



