import os

import torch
import numpy as np
from tqdm import tqdm
import pprint
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data
from criteria.loss import generator_loss_func, discriminator_loss_func


def train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda):
    # options
    device = torch.device(opts.device)
    opts_dict = OrderedDict(vars(opts))
    pprint.pprint(opts_dict)
    # data loader
    image_data_loader = sample_data(image_data_loader)
    pbar = range(opts.train_iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.01)
    
    if opts.distributed:
        generator_module, discriminator_module = generator.module, discriminator.module
    else:
        generator_module, discriminator_module = generator, discriminator
    
    writer = SummaryWriter(opts.log_dir)

    for index in pbar:
        
        i = index + opts.start_iter
        if i > opts.train_iter:
            print('Done...')
            break

        ground_truth, mask, edge, gray_image = next(image_data_loader)

        if is_cuda:
            ground_truth, mask, edge, gray_image = ground_truth.to(device), mask.to(device), edge.to(device), gray_image.to(device)

        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask

        # ---------
        # Generator
        # ---------
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        out_put = generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask)

        out_comp = ground_truth * mask + out_put * (1 - mask)

        output_pred, output_edge = discriminator(out_put, gray_image, edge, is_real=False)
        
        vgg_comp, vgg_output, vgg_ground_truth = extractor(out_comp), extractor(out_put), extractor(ground_truth)

        generator_loss_dict = generator_loss_func(
            mask, out_put, ground_truth, output_pred,
            vgg_comp, vgg_output, vgg_ground_truth,
            output_edge, edge
        )
        generator_loss = generator_loss_dict['loss_reconstruction'] * opts.RECONSTRUCTION_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS

        generator_loss_dict['loss_joint'] = generator_loss
        
        generator_optim.zero_grad()
        generator_loss.backward()
        generator_optim.step()

        # -------------
        # Discriminator
        # -------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        real_pred, real_pred_edge = discriminator(ground_truth, gray_image, edge, is_real=True)
        fake_pred, fake_pred_edge = discriminator(out_put.detach(), gray_image, edge, is_real=False)

        discriminator_loss_dict = discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge)
        discriminator_loss = discriminator_loss_dict['loss_adversarial']
        discriminator_loss_dict['loss_joint'] = discriminator_loss

        discriminator_optim.zero_grad()
        discriminator_loss.backward()
        discriminator_optim.step()

        # ---
        # log
        # ---
        generator_loss_dict_reduced, discriminator_loss_dict_reduced = reduce_loss_dict(generator_loss_dict), reduce_loss_dict(discriminator_loss_dict)

        pbar_g_loss_reconstruction = generator_loss_dict_reduced['loss_reconstruction'].mean().item()
        pbar_g_loss_perceptual = generator_loss_dict_reduced['loss_perceptual'].mean().item()
        pbar_g_loss_style = generator_loss_dict_reduced['loss_style'].mean().item()
        pbar_g_loss_adversarial = generator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_g_loss_joint = generator_loss_dict_reduced['loss_joint'].mean().item()

        pbar_d_loss_adversarial = discriminator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_d_loss_joint = discriminator_loss_dict_reduced['loss_joint'].mean().item()

        if get_rank() == 0:

            pbar.set_description((
                f'g_loss: {pbar_g_loss_joint:.4f}  '
                f'd_loss: {pbar_d_loss_joint:.4f}  '
                f'rec_loss: {pbar_g_loss_reconstruction * opts.RECONSTRUCTION_LOSS:.4f}  '
                f'per_loss: {pbar_g_loss_perceptual * opts.PERCEPTUAL_LOSS:.4f}  '
                f'style_loss: {pbar_g_loss_style * opts.STYLE_LOSS:.4f}  '
                f'adv_loss: {pbar_g_loss_adversarial * opts.ADVERSARIAL_LOSS:.4f}  '
            ))

            writer.add_scalar('g_loss_reconstruction', pbar_g_loss_reconstruction, i)
            writer.add_scalar('g_loss_perceptual', pbar_g_loss_perceptual, i)
            writer.add_scalar('g_loss_style', pbar_g_loss_style, i)
            writer.add_scalar('g_loss_adversarial', pbar_g_loss_adversarial, i)
            writer.add_scalar('g_loss_joint', pbar_g_loss_joint, i)

            writer.add_scalar('d_loss_adversarial', pbar_d_loss_adversarial, i)
            writer.add_scalar('d_loss_joint', pbar_d_loss_joint, i)

            if i % opts.save_interval == 0:
                torch.save(
                    {
                        'n_iter': i,
                        'generator': generator_module.state_dict(),
                        'discriminator': discriminator_module.state_dict()
                    },
                    f"{opts.save_dir}/{str(i).zfill(6)}.pt",
                )
            if i % opts.save_sample_iter == 0:
                input_image = (input_image + 1) * 127.5 / 255.
                ground_truth = (ground_truth + 1) * 127.5 / 255.
                out_comp = (out_comp + 1) * 127.5 / 255.
                compare = torch.cat((input_image.cpu(), out_comp.cpu(), ground_truth.cpu()), dim=0)
                save_image(compare, opts.sample_root + '/sample_{:05d}.png'.format(i), nrow=opts.batch_size)
