import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from models.EgoNeRF import EgoNeRF
import gc
import time


def volume_renderer(
        rays, model, chunk=4096, n_coarse=-1, n_fine=0, ndc_ray=False,
        white_bg=True, is_train=False, exp_sampling=False, device='cuda', empty_gpu_cache=False, pretrain_envmap=False,
        pivotal_sample_th=0., resampling=False, use_coarse_sample=True, interval_th=False
):
    if pretrain_envmap:
        env_map = model(rays_chunk=rays.to(device), pretrain_envmap=True)
        return env_map

    rgbs, alphas, depth_maps = [], [], []
    bg_maps = []
    env_maps = []
    N_rays_all = rays.shape[0]
    start = time.time()
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        if isinstance(model, EgoNeRF):
            rgb_map, depth_map, bg_map, env_map, alpha = model(
                rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, n_coarse=n_coarse, n_fine=n_fine,
                exp_sampling=exp_sampling, pivotal_sample_th=pivotal_sample_th, resampling=resampling,
                use_coarse_sample=use_coarse_sample, interval_th=interval_th
            )
        else:
            rgb_map, depth_map, bg_map, env_map, alpha = model(
                rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=n_coarse,
                exp_sampling=exp_sampling
            )

        if empty_gpu_cache:
            rgb_map_np = rgb_map.cpu().numpy()
            depth_map_np = depth_map.cpu().numpy()
            env_map_np = None
            alpha_np = alpha.cpu().numpy()
            if env_map is not None:
                bg_map_np = bg_map.cpu().numpy()
                env_map_np = env_map.cpu().numpy()
            rgbs.append(rgb_map_np)
            depth_maps.append(depth_map_np)
            alphas.append(alpha_np)
            if env_map is not None:
                bg_maps.append(bg_map_np)
                env_maps.append(env_map_np)
            del rgb_map, depth_map, bg_map, env_map, alpha
        else:
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
            if env_map is not None:
                bg_maps.append(bg_map)
                env_maps.append(env_map)
            alphas.append(alpha)
        if chunk_idx % 100 == 99:
            gc.collect()
            torch.cuda.empty_cache()


    if not empty_gpu_cache:
        if not is_train:
            print(f"elapsed time per image: {time.time() - start}")
        if env_map is not None:
            return torch.cat(rgbs), torch.cat(depth_maps), torch.cat(bg_maps), torch.cat(env_maps), torch.cat(alphas)
        else:
            return torch.cat(rgbs), torch.cat(depth_maps), None, None, torch.cat(alphas)
    else:
        if not is_train:
            print(f"elapsed time per image: {time.time() - start}")
        if env_map_np is not None:
            return np.concatenate(rgbs), np.concatenate(depth_maps), np.concatenate(bg_maps), np.concatenate(env_maps), np.concatenate(alphas)
        else:
            return np.concatenate(rgbs), np.concatenate(depth_maps), None, None, np.concatenate(alphas)


@torch.no_grad()
def evaluation(
        test_dataset, model, args, renderer, savePath=None, N_vis=5, prtx='', n_coarse=-1, n_fine=0,
        white_bg=False, ndc_ray=False, compute_extra_metrics=True, exp_sampling=False, device='cuda',
        empty_gpu_cache=False, envmap_only=False, resampling=False, use_coarse_sample=True, interval_th=False
):
    model.eval()
    # TODO: add WS-PSNR, WS-SSIM
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    if envmap_only:
        samples = test_dataset.all_rays[0]
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        env_map = renderer(rays, model, chunk=16384 * 4, device=device, pretrain_envmap=True)
        env_map = env_map.reshape(H, W, 3).cpu()
        env_map = (env_map.numpy() * 255).astype('uint8')
        imageio.imwrite(f'{savePath}/pretrained_envmap.png', env_map)
        return

    """
    if tensorf.envmap is not None:
        # envmap_img = tensorf.envmap.emission.detach().cpu().numpy()
        # envmap_img = torch.sigmoid(tensorf.envmap.emission.detach()).cpu().numpy()
        envmap_img = tensorf.envmap.get_radiance
        envmap_img = np.clip(envmap_img, 0, 1)
        envmap_img = envmap_img.transpose(2, 1, 0)
        envmap_img = (envmap_img * 255).astype(np.uint8)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}envmap.png', envmap_img)
    """
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        rgb_map, depth_map, bg_map, env_map, _ = renderer(
            rays, model, chunk=4096, n_coarse=n_coarse, n_fine=n_fine, ndc_ray=ndc_ray, white_bg=white_bg,
            exp_sampling=exp_sampling, device=device, empty_gpu_cache=empty_gpu_cache, resampling=resampling,
            use_coarse_sample=use_coarse_sample, interval_th=interval_th
        )
        if empty_gpu_cache:
            rgb_map = rgb_map.clip(0., 1.)
            rgb_map, depth_map = torch.from_numpy(rgb_map.reshape(H, W, 3)), torch.from_numpy(depth_map.reshape(H, W))
            if env_map is not None:
                bg_map = torch.from_numpy(bg_map.reshape(H, W, 3))
                env_map = torch.from_numpy(env_map.reshape(H, W, 3))
        else:
            rgb_map = rgb_map.clamp(0.0, 1.0)
            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
            if env_map is not None:
                bg_map = bg_map.reshape(H, W, 3).cpu()
                if idx == 0:
                    env_map = env_map.reshape(H, W, 3).cpu()

        if env_map is not None:
            bg_map = (bg_map.numpy() * 255).astype('uint8')
            if idx == 0:
                env_map = (env_map.numpy() * 255).astype('uint8')

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', model.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', model.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
            if env_map is not None:
                if idx == 0:
                    imageio.imwrite(f'{savePath}/{prtx}envmap.png', env_map)
                imageio.imwrite(f'{savePath}/{prtx}{idx:03d}_bg.png', bg_map)
    gc.collect()
    torch.cuda.empty_cache()

    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))
    model.train()

    return PSNRs


@torch.no_grad()
def evaluation_path(test_dataset,model, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, exp_sampling=False, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, depth_map, _, _, _ = renderer(
            rays, model, chunk=8192, N_samples=N_samples, ndc_ray=ndc_ray, white_bg=white_bg,
            exp_sampling=exp_sampling, device=device
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs
