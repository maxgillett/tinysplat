import logging
from argparse import ArgumentParser

import torch

from tinysplat.model import GaussianModel

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s')

    parser = ArgumentParser(description='Configuration parameters')
    parser.add_argument('--filetype', type=str, default='PLY')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    # Load model from checkpoint
    state_dict = torch.load(args.input_file)
    model = GaussianModel.from_state_checkpoint(state_dict)

    # Export model to PLY or SPLAT
    if args.filetype == 'PLY':
        model.export_ply(args.output_file)
    elif args.filetype == 'SPLAT':
        model.export_splat(args.output_file)
    else:
        raise ValueError('Unknown filetype: {}'.format(args.filetype))

if __name__ == "__main__":
    main()
