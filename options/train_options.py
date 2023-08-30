import argparse
import pprint
from collections import OrderedDict
from options.base_options import BaseOptions

class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()
        # ------------------
        # dataset parameters
        # ------------------
        self.parser.add_argument('--image_root', type=str, default='')
        self.parser.add_argument('--mask_root', type=str, default='')
        self.parser.add_argument('--sample_root', type=str, default='./sample')
        self.parser.add_argument('--save_dir', type=str, default='checkpoints/ckpt_celeba')
        self.parser.add_argument('--log_dir', type=str, default='runs/log_celeba')
        self.parser.add_argument('--pre_trained', type=str, default='')

        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--load_size', type=int, default=(256, 256))
        self.parser.add_argument('--sigma', type=float, default=2.)
        self.parser.add_argument('--mode', type=str, default='train')

        self.parser.add_argument('--gen_lr', type=float, default=0.0002)
        self.parser.add_argument('--D2G_lr', type=float, default=0.1)
        self.parser.add_argument('--lr_finetune', type=float, default=0.00005)
        self.parser.add_argument('--finetune', type=bool, default=False)
        
        self.parser.add_argument('--start_iter', type=int, default=1)
        self.parser.add_argument('--train_iter', type=int, default=300000)
        self.parser.add_argument('--save_interval', type=int, default=10000)
        self.parser.add_argument('--save_sample_iter', type=int, default=1000)

        self.parser.add_argument('--RECONSTRUCTION_LOSS', type=float, default=8.0)
        self.parser.add_argument('--PERCEPTUAL_LOSS', type=float, default=0.1)   # defualt 0.1
        self.parser.add_argument('--STYLE_LOSS', type=float, default=250.)   # default 250
        self.parser.add_argument('--ADVERSARIAL_LOSS', type=float, default=0.1)

        # -----------
        # Distributed
        # -----------
        self.parser.add_argument('--local_rank', type=int, default=0, help="local rank for distributed training")

        self.opts = self.parser.parse_args()

    @property
    def parse(self):

        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts
