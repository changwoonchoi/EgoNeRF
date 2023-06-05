import configargparse
import os
from pathlib import Path


def load_all_include(config_file):
    parser = config_parser()
    args = parser.parse_args("--config {}".format(config_file))
    path = Path(config_file)

    include = []
    if args.include:
        include.append(os.path.join(path.parent, args.include))
        return include + load_all_include(os.path.join(path.parent, args.include))
    else:
        return include


def recursive_config_parser():
    parser = config_parser()
    args = parser.parse_args()
    include_files = load_all_include(args.config)
    include_files = list(reversed(include_files))
    parser = config_parser(default_files=include_files)
    return parser


def config_parser(default_files=None):
    if default_files is not None:
        parser = configargparse.ArgumentParser(default_config_files=default_files)
    else:
        parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--include', type=str, default=None, help='parent config file path')

    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log', help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0, help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--test_skip', type=int, default=1, help='skip test set for fast visualization')

    parser.add_argument('--model_name', type=str, default='EgoNeRF',
                        choices=['TensorVMSplit', 'TensorCP', 'EgoNeRF'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='omniblender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'matterport', 'egocentric', 'omniblender', 'omniscenes', 'openvslam', 'own_data'])
    parser.add_argument('--localization_method', type=str, default='colmap',
                        choices=['colmap', 'openvslam', 'pix4d'])
    parser.add_argument('--near_far', type=float, action='append')
    parser.add_argument('--roi', type=float, action='append')

    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.005, help='learning rate')
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr_envmap_pretrain", type=float, default=0.02, help='learning rate for envmap')
    parser.add_argument("--lr_envmap", type=float, default=0.005, help='learning rate for envmap')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help='number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_initial", type=float, default=0.0, help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0, help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0, help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help='loss weight')
    parser.add_argument("--entropy_weight", type=float, default=0.0, help='weight of ray entropy loss')
    parser.add_argument("--iter_ignore_entropy", type=int, default=0, help='ignore ray entropy loss for the first few iterations')
    parser.add_argument("--iter_ignore_TV", type=int, default=1e5, help='ignore TV loss after the first few iterations')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    
    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--evaluation", type=int, default=0, help='render and evaluate test set')
    parser.add_argument("--metric_only", type=int, default=0, help='evaluate metrics in test set from existing rendered images')

    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)
    parser.add_argument('--exp_sampling', default=False, action="store_true",
                        help='exponential sampling')
    parser.add_argument('--resampling', default=False, action='store_true')
    parser.add_argument('--n_coarse', type=int, default=128, help='number of coarse samples along camera ray')
    parser.add_argument('--n_fine', type=int, default=64, help='number of fine samples to resample from coarse ray weights')
    parser.add_argument('--ray_weight_th', type=float, default=0.01, help='ray weight threshold value for filtering pivot coarse samples')
    parser.add_argument('--use_coarse_sample', action='store_true', help='use both coarse samples and fine samples for rendering')

    # coarse sigma grid options
    parser.add_argument("--coarse_sigma_grid_update_rule", type=str, default=None, choices=["conv", "samp"],
                        help="coarse sigma grid updating strategy. "
                             "conv: obtain coarse sigma grid by convolving with a kernel. "
                             "samp: obtain coarse sigma grid by sampling from fine sigma grid.")
    parser.add_argument("--pivotal_sample_th", type=float, default=0., help="weight threshold value for filtering coarse samples to obtain pivotal samples")
    parser.add_argument("--iter_ignore_resampling", type=int, default=-1, help="ignore resampling for the first few iterations")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")
    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')


    # envmap options
    parser.add_argument("--use_envmap", default=False, action="store_true")
    parser.add_argument("--envmap_res_H", type=int, default=1000)
    parser.add_argument("--iter_pretrain_envmap", type=int, default=0)

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument("--filter_ray", action='store_true', help='filter rays in bbox')
    parser.add_argument('--N_voxel_init', type=int, default=100**3)
    parser.add_argument('--N_voxel_final', type=int, default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")

    parser.add_argument('--idx_view', type=int, default=0)

    # logging/saving options.
    parser.add_argument("--N_vis", type=int, default=-1, help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000, help='frequency of visualize the image, deprecated!!')
    parser.add_argument("--vis_list", type=int, action="append", help='list of visualization steps')
    parser.add_argument("--i_weights", type=int, default=5000, help='frequency of save the weights')

    # depth supervision
    parser.add_argument("--use_depth", action='store_true',
                        help="use depth supervision.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help='depth lambda for loss')
    parser.add_argument("--depth_step_size", type=int, default=5000,
                        help='reducing depth every')
    parser.add_argument("--depth_rate", type=float, default=1,
                        help='reducing depth rate')
    parser.add_argument("--depth_end_iter", type=int,
                        help='when smoothing will be end')    
    parser.add_argument("--use_gt_depth", action='store_true',
                        help='use ground truth depth value')

    # coordinates
    parser.add_argument('--coordinates_name', type=str, default='xyz',
                        choices=['xyz', 'sphere', 'cylinder', 'balanced_sphere', 'directional_sphere', 'directional_balanced_sphere', 'euler_sphere', 'yinyang', 'generic_sphere'])
    parser.add_argument('--r0', type=float, default=None, help='radius of initial spherical shell')
    parser.add_argument('--interval_th', action='store_true', help='force minimum r-grid interval to be r0')

    # sparsity loss
    parser.add_argument("--sparsity_lambda", type=float, default=0.1,
                        help='sparsity lambda for loss')
    parser.add_argument("--N_sparsity_points", type=int, default=10000,
                        help='N points to sample for sparsity loss calculation')
    parser.add_argument("--sparsity_length", type=float, default=0.2,
                        help='hyper param for sparse alpha composition')

    # Sampler
    parser.add_argument('--sampling_method', type=str, default='simple',
                        choices=['simple', 'brute_force', 'theta_importance'])
    parser.add_argument('--theta_importance_lambda', type=float, default=5,
                        help='weight for theta importance sampling')

    return parser


def export_config(args, logdir):
    """
    Create log dir and copy the config file
    """
    f = os.path.join(logdir, 'args.txt')
    with open(f, 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(logdir, 'config.txt')
        with open(f, 'w') as f:
            f.write(open(args.config, 'r').read())
