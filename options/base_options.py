import argparse
import pprint
from collections import OrderedDict

class BaseOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--in_channels', type=int, default=3, help='input RGB image')
        self.parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
        self.parser.add_argument('--edge_in_channels', type=int, default=2, help='input edge image')
        self.parser.add_argument('--latent_channels', type=int, default=32, help='latent channels')
        self.parser.add_argument('--device', type=str, default='cpu', help='device for experiment')

        self.opt = self.parser.parse_args()

    def parse(self):
        opt_dict = OrderedDict(vars(self.opt))
        # pprint.pprint(opt_dict)
        return self.opt