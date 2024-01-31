import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    args.config = f"./exps/{args.model_name}.json"
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    param.update(args)
    train(param)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')

    parser.add_argument('--dataset', type=str, default="etcdata")
    parser.add_argument('--init_cls', '-init', type=int, default=2)
    parser.add_argument('--increment', '-incre', type=int, default=2)
    parser.add_argument('--model_name','-model', type=str, default='CLCR', required=True)
    parser.add_argument('--convnet_type','-net', type=str, default='CLCR_Classfier')
    parser.add_argument('--device','-d', nargs='+', type=int, default=[0,1,2,3])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip', action="store_true",)
    
    return parser


if __name__ == '__main__':
    main()