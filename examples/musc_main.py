import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.musc import MuSc
from utils.load_config import load_yaml

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='MuSc')
    parser.add_argument('--config', type=str, default='./configs/musc.yaml', help='config file path')
    parser.add_argument('--data_path', type=str, default=None, help='dataset path')
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
    parser.add_argument('--class_name', type=str, default=None, help='category')
    parser.add_argument('--device', type=int, default=None, help='gpu id')
    parser.add_argument('--output_dir', type=str, default=None, help='save results path')
    parser.add_argument('--vis', type=str, default=None, help='visualization')
    parser.add_argument('--vis_type', type=str, default=None, help='normalization type in visualization')
    parser.add_argument('--save_excel', type=str, default=None, help='save excel')
    parser.add_argument('--r_list', type=int, nargs="+", default=None, help='aggregation degrees of LNAMD')
    parser.add_argument('--feature_layers', type=int, nargs="+", default=None, help='feature layers')
    parser.add_argument('--backbone_name', type=str, default=None, help='backbone')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained datasets')
    parser.add_argument('--img_resize', type=int, default=None, help='image size')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--divide_num', type=int, default=None, help='the number of divided subsets')
    args = parser.parse_args()
    return args

def load_args(cfg, args):
    # If input new arguments through the script (musc.sh), the default configuration in the config file (musc.yaml) will be overwritten.
    if args.data_path is not None:
        cfg['datasets']['data_path'] = args.data_path
    assert os.path.exists(cfg['datasets']['data_path']), f"The dataset path {cfg['datasets']['data_path']} does not exist."
    if args.dataset_name is not None:
        cfg['datasets']['dataset_name'] = args.dataset_name
    if args.class_name is not None:
        cfg['datasets']['class_name'] = args.class_name
    if args.device is not None:
        cfg['device'] = args.device
    if isinstance(cfg['device'], int):
        cfg['device'] = str(cfg['device'])
    if args.output_dir is not None:
        cfg['testing']['output_dir'] = args.output_dir
    os.makedirs(cfg['testing']['output_dir'], exist_ok=True)
    if args.vis is not None:
        if args.vis.lower() == 'true':
            cfg['testing']['vis'] = True
        else:
            cfg['testing']['vis'] = False
    if args.vis_type is not None:
        cfg['testing']['vis_type'] = args.vis_type
    if args.save_excel is not None:
        if args.save_excel.lower() == 'true':
            cfg['testing']['save_excel'] = True
        else:
            cfg['testing']['save_excel'] = False
    if args.r_list is not None:
        cfg['models']['r_list'] = args.r_list
    if isinstance(cfg['models']['r_list'], int):
        cfg['models']['r_list'] = [cfg['models']['r_list']]
    if args.feature_layers is not None:
        cfg['models']['feature_layers'] = args.feature_layers
    if isinstance(cfg['models']['feature_layers'], int):
        cfg['models']['feature_layers'] = [cfg['models']['feature_layers']]
    if args.backbone_name is not None:
        cfg['models']['backbone_name'] = args.backbone_name
    if args.pretrained is not None:
        cfg['models']['pretrained'] = args.pretrained
    if args.img_resize is not None:
        cfg['datasets']['img_resize'] = args.img_resize
    if args.batch_size is not None:
        cfg['models']['batch_size'] = args.batch_size
    if args.divide_num is not None:
        cfg['datasets']['divide_num'] = args.divide_num
    return cfg

if __name__ == "__main__":
    args = get_args()
    cfg = load_yaml(args.config)
    cfg = load_args(cfg, args)
    print(cfg)
    seed = 42
    model = MuSc(cfg, seed=seed)
    model.main()



