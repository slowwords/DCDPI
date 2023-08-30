from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.generator.generator import Generator
import warnings
from scripts.lpips import LPIPS
import lpips
warnings.filterwarnings("ignore")
from torchvision.utils import save_image
import os
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from datasets.dataset import create_image_dataset
from options.test_options import TestOptions
from utils.misc import sample_data, postprocess
from thop import profile
is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

opts = TestOptions().parse

device = torch.device(opts.device)

def load_model(path, generator, device):
    model_dict = torch.load(path, map_location=device)
    # generator
    G_dict = generator.state_dict()
    G_pre_dict = model_dict['generator']
    G_pred_dict = {k: v for k, v in G_pre_dict.items() if k in G_dict}
    G_dict.update(G_pred_dict)
    generator.load_state_dict(G_dict, strict=False)
    print(f"load pretrained G weights")


def test():
    # model & load model
    generator = Generator(opts)
    if opts.pre_trained != '':
        if opts.device == "cpu":
            load_model(opts.pre_trained, generator)
            generator.load_state_dict(torch.load(opts.pre_trained, map_location="cpu")['generator'])
        else:
            generator.load_state_dict(torch.load(opts.pre_trained)['generator'])
    else:
        print('Please provide pre-trained model!')

    if is_cuda:
        generator = generator.to(device)

    # dataset
    image_dataset = create_image_dataset(opts)
    image_data_loader = data.DataLoader(
        image_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        drop_last=False
    )
    image_data_loader = sample_data(image_data_loader)
    print('start eval...')
    with torch.no_grad():
        generator.eval()

        for _ in tqdm(range(opts.number_eval)):

            ground_truth, mask, edge, gray_image = next(image_data_loader)
            if is_cuda:
                ground_truth, mask, edge, gray_image = ground_truth.to(device), mask.to(device), edge.to(device), gray_image.to(device)

            input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask
            output, ec_image, ec_edge, dc3, dc2 = generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask)
            output_comp = ground_truth * mask + output * (1 - mask)

            output_comp = postprocess(output_comp)

            save_image(output_comp, opts.result_root + '/{:05d}.png'.format(_))


if __name__ == '__main__':
    test()
