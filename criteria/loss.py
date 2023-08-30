import torch
import torch.nn as nn

from utils.misc import gram_matrix
from options.base_options import BaseOptions
opt = BaseOptions().parse()
device = torch.device(opt.device)

def generator_loss_func(
    mask, out_put, ground_truth, output_pred,
    vgg_comp, vgg_output, vgg_ground_truth,
    output_edge, edge):

    l1 = nn.L1Loss()
    criterion = nn.BCELoss()


    # ---------
    # hole loss
    # ---------
    loss_hole = Charbonnier_loss((1 - mask) * out_put, (1 - mask) * ground_truth)  # shape=(1, 3, 256, 256)

    # ----------
    # valid loss
    # ----------
    loss_valid = Charbonnier_loss(mask * out_put, mask * ground_truth)

    loss_reconstruction = 6 * loss_hole + loss_valid

    # ---------------
    # perceptual loss
    # ---------------
    loss_perceptual = 0.0
    for i in range(3):
        loss_perceptual += l1(vgg_output[i], vgg_ground_truth[i])
        loss_perceptual += l1(vgg_comp[i], vgg_ground_truth[i])

    # ----------
    # style loss
    # ----------
    loss_style = 0.0
    for i in range(3):
        loss_style += l1(gram_matrix(vgg_output[i]), gram_matrix(vgg_ground_truth[i]))
        loss_style += l1(gram_matrix(vgg_comp[i]), gram_matrix(vgg_ground_truth[i]))

    # ----------------
    # adversarial loss
    # ----------------
    real_target = torch.tensor(1.0).expand_as(output_pred)
    if torch.cuda.is_available():
        real_target = real_target.to(device)
    loss_adversarial = criterion(output_pred, real_target) + criterion(output_edge, edge)

    
    # -----------------
    # intermediate loss
    # -----------------
    # loss_intermediate = criterion(projected_edge, edge) + l1(projected_image, ground_truth)

    return {
        'loss_reconstruction': loss_reconstruction.mean(),
        'loss_perceptual': loss_perceptual.mean(), 
        'loss_style': loss_style.mean(), 
        'loss_adversarial': loss_adversarial.mean(),
    }


def discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge):

    criterion = nn.BCELoss()
    
    real_target = torch.tensor(1.0).expand_as(real_pred)
    fake_target = torch.tensor(0.0).expand_as(fake_pred)
    if torch.cuda.is_available():
        real_target = real_target.to(device)
        fake_target = fake_target.to(device)

    loss_adversarial = criterion(real_pred, real_target) + criterion(fake_pred, fake_target) +\
                       criterion(real_pred_edge, edge) + criterion(fake_pred_edge, edge)

    return {
        'loss_adversarial': loss_adversarial.mean()
    }


def Charbonnier_loss(X, Y, eps=1e-6):
    diff = torch.add(X, - Y)
    error = torch.sqrt(diff * diff + eps)   # eps为正则项
    loss = torch.mean(error)
    return loss