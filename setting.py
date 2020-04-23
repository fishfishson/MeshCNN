'''
Configs for training & testing
Written by Whalechen
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lst_path', default='datasets/train_list.txt', help='path to dataset file list')
    parser.add_argument('--dataroot', default='datasets', help='path to model data dir')
    parser.add_argument('--dataset_mode', choices={"classification", "segmentation"}, default='segmentation')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 64])
    parser.add_argument('--ninput_edges', type=int, default=3000,
                        help='# of input edges (will include dummy edges)')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples per epoch')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')

    # network params
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--arch', type=str, default='meshunet', help='selects network to use')  # todo add choices
    parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
    parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses')  # todo make generic
    parser.add_argument('--input_nc', type=int, default=100, help='# between fc and nclasses')  # todo make generic
    parser.add_argument('--ncf', nargs='+', default=[64, 128, 256, 256], type=int, help='conv filters')
    parser.add_argument('--pool_res', nargs='+', default=[2000, 1000, 500], type=int, help='pooling res')
    parser.add_argument('--norm', type=str, default='batch',
                        help='instance normalization or batch normalization or group normalization')
    parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
    parser.add_argument('--init_type', type=str, default='kaiming',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--nclasses', type=int, default=2, help='# of classes')
    parser.add_argument('--seg_inplanes', type=int, default=16, help='# of seg net inplanes')

    # general params
    parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--name', type=str, default='debug',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--resume_path', default='', type=str, help='Path for resume model.')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes meshes in order, otherwise takes them randomly')
    parser.add_argument('--seed', default=1, type=int, help='Manually set random seed')

    # visualization params
    parser.add_argument('--export_folder', type=str, default='',
                        help='exports intermediate collapses to this folder')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=250,
                        help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=1,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--run_test_freq', type=int, default=1,
                        help='frequency of running test in training script')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--nepoch', type=int, default=1000)
    parser.add_argument('--save_intervals', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./datasets')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--which_epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=2000,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='step',
                        help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')

    # data augmentation stuff
    parser.add_argument('--num_aug', type=int, default=10, help='# of augmentation files')
    parser.add_argument('--scale_verts', action='store_true',
                        help='non-uniformly scale the mesh e.g., in x, y or z')
    parser.add_argument('--slide_verts', type=float, default=0,
                        help='percent vertices which will be shifted along the mesh surface')
    parser.add_argument('--flip_edges', type=float, default=0, help='percent of edges to randomly flip')

    # tensorboard visualization
    parser.add_argument('--no_vis', action='store_true', help='will not use tensorboard')
    parser.add_argument('--verbose_plot', action='store_true', help='plots network weights, etc.')

    args = parser.parse_args()
    args.save_folder = "./trails/models/{}_{}".format(args.model, args.model_depth)

    return args
