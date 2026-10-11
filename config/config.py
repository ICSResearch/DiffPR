import argparse
import torch

parser = argparse.ArgumentParser(description='parameters of the program')
parser.add_argument("--rate", default=4, type=int)
parser.add_argument("--alpha", default=9, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--measurement_type", default="CDPs")
parser.add_argument('--device', default=0, required=True, help='device index')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument("--patch_size", default=128, type=int)
parser.add_argument("--time_steps", default=7, type=int)

parser.add_argument('--save_dir', default='./trained_models', help='save path of results')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--pretrained', default=False, type=bool, help='is pretrained or not')
parser.add_argument("--save", default=False)
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument("--reconstruct_results", default='./reconstructed_images', type=str, help='directory to save reconstructed images')

para = parser.parse_args()
if torch.cuda.is_available():
    para.device = f"cuda:{para.device}"
else:
    para.device = "cpu"

para.dataset_root = 'data'





