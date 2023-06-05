import gc
import os, sys
from tqdm.auto import tqdm
from opt import recursive_config_parser, export_config

from renderer import volume_renderer, evaluation
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from models.EgoNeRF import EgoNeRF

from dataLoader import dataset_dict
from models import coordinates_dict

from sampler import SimpleSampler, ThetaImportanceSampler
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def render_test(args):
    if args.metric_only:
        raise NotImplementedError
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    load_params = {
        "data_dir": args.datadir,
        "split": "test",
        "is_stack": True,
        "use_gt_depth": args.use_gt_depth,
        "downsample": 1,
        "near_far": args.near_far,
        "roi": args.roi,
        "localization_method": args.localization_method,
        "skip": 1,
    }
    test_dataset = dataset(**load_params)
    white_bg = test_dataset.white_bg

    if args.ckpt is None:
        ckpt = os.path.join(f'{args.basedir}/{args.expname}/{args.expname}.th')
    else:
        ckpt = args.ckpt

    if not os.path.exists(ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    model = eval(args.model_name)(**kwargs)
    model.load(ckpt)
    renderer = volume_renderer

    if args.ckpt is None:
        evaluation_dir = f'{args.basedir}/{args.expname}/evaluation'
    else:
        evaluation_dir = f'{args.basedir}/{args.expname}/evaluation_{args.ckpt.split(".")[-2].split("_")[-1]}'
    os.makedirs(evaluation_dir, exist_ok=True)

    evaluation(
        test_dataset, model, args, renderer, evaluation_dir, N_vis=-1, n_coarse=args.n_coarse, n_fine=args.n_fine,
        white_bg=white_bg, ndc_ray=False, device=device, exp_sampling=args.exp_sampling, compute_extra_metrics=True,
        resampling=args.resampling, empty_gpu_cache=True, use_coarse_sample=args.use_coarse_sample, interval_th=args.interval_th
    )


def train(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    load_params = {
        "data_dir": args.datadir,
        "split": "train",
        "is_stack": False,
        "use_gt_depth": args.use_gt_depth,
        "downsample": args.downsample_train,
        "near_far": args.near_far,
        "localization_method": args.localization_method,
    }
    train_dataset = dataset(**load_params)

    load_params["split"] = "test"
    load_params["is_stack"] = True
    load_params["downsample"] = args.downsample_test
    load_params["skip"] = args.test_skip  # only test dataset can have non one skip value
    test_dataset = dataset(**load_params)

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    vis_list = args.vis_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    export_config(args, logfolder)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)

    # init coordinates
    coordinates = coordinates_dict[args.coordinates_name]
    is_yinyang = args.coordinates_name == 'yinyang'
    # for backward compatibility
    if is_yinyang:
        coordinates = coordinates(
            device, aabb, exp_r=args.exp_sampling, N_voxel=args.N_voxel_init, r0=args.r0, interval_th=args.interval_th
        )
    else:
        coordinates = coordinates(device, aabb)

    reso_cur = coordinates.N_to_reso(args.N_voxel_init, aabb)
    if not is_yinyang:
        coordinates.set_resolution(reso_cur)

    n_coarse = args.n_coarse
    n_fine = args.n_fine if args.resampling else 0
    use_coarse_sample = args.use_coarse_sample
    if args.resampling:
        coarse_sigma_grid_update_rule = args.coarse_sigma_grid_update_rule
        if coarse_sigma_grid_update_rule == "conv":
            coarse_sigma_grid_update_step = 1
        elif coarse_sigma_grid_update_rule == "samp":
            raise NotImplementedError
        else:
            coarse_sigma_grid_update_step = 1000000000
    else:
        coarse_sigma_grid_update_rule = None
        coarse_sigma_grid_update_step = 1000000000

    renderer = volume_renderer

    ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.th')]
    if args.ckpt is not None or len(ckpts) > 0:
        if args.ckpt is None:
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'\n\nload ckpt from {ckpt_path}!!\n\n')
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        model = eval(args.model_name)(**kwargs)
        start = model.load(ckpt)
    else:
        start = 0
        model = eval(args.model_name)(
            aabb, reso_cur, device, coordinates, density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color, near_far=near_far, shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift,
            distance_scale=args.distance_scale, pos_pe=args.pos_pe, view_pe=args.view_pe,
            fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct, use_envmap=args.use_envmap, envmap_res_H=int(args.envmap_res_H / args.downsample_train,),
            coarse_sigma_grid_update_rule=coarse_sigma_grid_update_rule, coarse_sigma_grid_reso=None, interval_th=args.interval_th
        )
    if args.iter_pretrain_envmap > 0:
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap_pretrain)
    else:
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    lr_factor = lr_factor ** start

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logarithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]
    valid_sample_ratios = []

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if args.use_depth:
        alldepths = train_dataset.all_depths

    if args.sampling_method == 'simple':
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    elif args.sampling_method == 'theta_importance':
        img_len = allrays.shape[0] // np.prod(train_dataset.img_wh)
        trainingSampler = ThetaImportanceSampler(args.theta_importance_lambda, img_len, train_dataset.img_wh,
                                                 args.batch_size, train_dataset.roi)
    else:
        raise ValueError('sampling method not supported')

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_initial
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    entropy_weight = args.entropy_weight
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    if args.use_envmap and args.iter_pretrain_envmap > 0:
        pbar_pretrain = tqdm(range(args.iter_pretrain_envmap), miniters=50, file=sys.stdout)
        print("\n\n pretrain envmap")

        for pretrain_iter in pbar_pretrain:
            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
            env_map = renderer(
                rays_train, model, chunk=16384 * 4, device=device, is_train=True, pretrain_envmap=True
            )
            loss_pretrain_envmap = torch.mean((env_map - rgb_train) ** 2)
            optimizer.zero_grad()
            loss_pretrain_envmap.backward()
            optimizer.step()
            if pretrain_iter % 50 == 49:
                pbar_pretrain.set_description(f'Iteration {pretrain_iter:04d}: {loss_pretrain_envmap.item()}')

        evaluation(
            test_dataset, model, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
            n_coarse=0, white_bg=white_bg, ndc_ray=False, compute_extra_metrics=False,
            exp_sampling=args.exp_sampling, empty_gpu_cache=True, envmap_only=True
        )
        # reset lr rate of envmap
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap)
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    pbar = tqdm(range(start, args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        if args.use_depth:
            depth_train = alldepths[ray_idx].to(device).squeeze()
            depth_nonezero_mask = depth_train != 0

        rgb_map, depth_map, _, _, alpha = renderer(
            rays_train, model, chunk=args.batch_size, n_coarse=n_coarse,
            n_fine=n_fine, white_bg=white_bg, ndc_ray=False, device=device, is_train=True,
            exp_sampling=args.exp_sampling, pivotal_sample_th=args.pivotal_sample_th,
            resampling=(args.resampling and iteration > args.iter_ignore_resampling), use_coarse_sample=use_coarse_sample,
            interval_th=args.interval_th
        )

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss
        total_loss = loss

        # sparsity loss (ref: DirectVoxGO)
        if args.sparsity_lambda > 0:
            sample_points = torch.rand((args.N_sparsity_points, 3), device=device) * 2 - 1
            sp_sigma = model.compute_densityfeature(sample_points)
            sp_sigma = model.feature2density(sp_sigma)
            loss_sp = 1.0 - torch.exp(-args.sparsity_length * sp_sigma).mean()

            total_loss = total_loss + args.sparsity_lambda * loss_sp

        # depth loss calculation
        depth_lambda = args.depth_lambda * args.depth_rate ** (int(iteration / args.depth_step_size))
        if args.use_depth:
            depth_loss = torch.mean((depth_map[depth_nonezero_mask] - depth_train[depth_nonezero_mask]) ** 2)
            if args.depth_end_iter is not None:
                if iteration > args.depth_end_iter:
                    depth_loss = 0

            total_loss = total_loss + depth_lambda * depth_loss

        if Ortho_reg_weight > 0:
            loss_reg = model.vector_comp_diffs()
            total_loss = total_loss + Ortho_reg_weight * loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)

        if L1_reg_weight > 0:
            loss_reg_L1 = model.density_L1()
            total_loss = total_loss + L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density > 0 and iteration < args.iter_ignore_TV:
            TV_weight_density *= lr_factor
            loss_tv = model.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0 and iteration < args.iter_ignore_TV:
            TV_weight_app *= lr_factor
            loss_tv = model.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        if entropy_weight > 0 and iteration > args.iter_ignore_entropy:
            entropy_weight *= lr_factor
            loss_entropy = ray_entropy_loss(alpha)
            total_loss = total_loss + loss_entropy * entropy_weight
            summary_writer.add_scalar('train/ray_entropy_loss', loss_entropy.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        # valid_sample_ratios.append(valid_sample_ratio.detach().item())
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)
        # summary_writer.add_scalar('train/valid_sample_ratio', valid_sample_ratios[-1], global_step=iteration)

        if iteration % 1000 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
                # + f' valid_sample_ratio = {float(np.mean(valid_sample_ratios)):.2f}'
            )
            PSNRs = []

        # if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
        if (iteration + 1) in vis_list and args.N_vis != 0:
            PSNRs_test = evaluation(
                test_dataset, model, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                prtx=f'{iteration:06d}_', n_coarse=n_coarse, n_fine=n_fine, white_bg=white_bg, ndc_ray=False,
                compute_extra_metrics=False, exp_sampling=args.exp_sampling, empty_gpu_cache=True,
                resampling=(args.resampling and iteration > args.iter_ignore_resampling), use_coarse_sample=use_coarse_sample,
                interval_th=args.interval_th
            )
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration % args.i_weights == 0 and iteration != 0:
            model.save(f'{logfolder}/{args.expname}_{iteration:06d}.th', global_step=iteration)

        if args.resampling and (iteration + 1) % coarse_sigma_grid_update_step == 0 and is_yinyang:
            model.update_coarse_sigma_grid()

        if update_AlphaMask_list and iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] <= 128 ** 3:  # update volume resolution
                reso_mask = reso_cur
            new_aabb = model.updateAlphaMask(tuple(reso_mask))
            # new_aabb = tensorf.updateAlphaMask()
            if iteration == update_AlphaMask_list[0]:
                # tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            # reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            reso_cur = coordinates.N_to_reso(n_voxels, model.aabb)
            # nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            model.upsample_volume_grid(reso_cur)
            coordinates.set_resolution(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = model.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale, args.lr_envmap * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    model.save(f'{logfolder}/{args.expname}.th', global_step=iteration)

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_train = evaluation(
            train_dataset, model, args, renderer, f'{logfolder}/imgs_train_all/', N_vis=-1, N_samples=n_coarse,
            white_bg=white_bg, ndc_ray=False, device=device, exp_sampling=args.exp_sampling
        )
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(
            test_dataset, model, args, renderer, f'{logfolder}/imgs_test_all/', N_vis=-1, n_coarse=n_coarse,
            n_fine=n_fine, white_bg=white_bg, ndc_ray=False, device=device, exp_sampling=args.exp_sampling,
            empty_gpu_cache=True, resampling=args.resampling, use_coarse_sample=use_coarse_sample
        )
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20221028)
    np.random.seed(20221028)

    parser = recursive_config_parser()
    args = parser.parse_args()
    print_arguments(args)

    if args.evaluation:
        render_test(args)
    else:
        train(args)
