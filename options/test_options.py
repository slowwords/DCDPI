import argparse
import pprint
from collections import OrderedDict
from options.base_options import BaseOptions

class TestOptions(BaseOptions):

    def __init__(self):
        super(TestOptions, self).__init__()
        # ------------------
        # dataset parameters
        # ------------------
        self.parser.add_argument('--pre_trained', type=str,
                            default='')
        self.parser.add_argument('--image_root', type=str,
                            default='')
        self.parser.add_argument('--mask_root', type=str,
                            default='')
        self.parser.add_argument('--result_root', type=str,
                            default='')

        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--load_size', type=int, default=(256, 256))
        self.parser.add_argument('--sigma', type=float, default=2.)
        self.parser.add_argument('--mode', type=str, default='test')

        self.parser.add_argument('--number_eval', type=int, default=10)

        self.opts = self.parser.parse_args()

    @property
    def parse(self):

        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts
