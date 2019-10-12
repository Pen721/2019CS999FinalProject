import argparse, os, torch
from DRAGAN import DRAGAN
from WassersteinGAN import WGAN

def parse_args():
    desc = "CS999"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gan_type', type=str, default='DRAGAN', choices= ['DRAGAN', 'WGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=8, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    #parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    return check_args(parser.parse_args())

def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args

def main():
    # arguments
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    if args.gan_type == 'DRAGAN':
        gan = DRAGAN(args)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args)
    gan.train()
    #visualization
    gan.visualize_results(args.epoch)

#run
if __name__ == '__main__':
    main()
