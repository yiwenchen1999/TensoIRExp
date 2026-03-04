"""
Batch training + relighting pipeline for polyhaven_lvsm dataset.

For each relight_metadata JSON:
  1) Train TensoIR on scene_name
  2) Load the envmap from relit_scene_name
  3) Render relit views for target_view_indices
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import imageio
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opt import config_parser
from renderer import Renderer_TensoIR_train
from models.tensoRF_rotated_lights import TensorVMSplit, AlphaGridMask
from models.relight_utils import (
    GGX_specular, linear2srgb_torch, compute_transmittance,
)
from dataLoader.ray_utils import safe_l2_normalize
from dataLoader import dataset_dict
from utils import N_to_reso, cal_n_samples, TVLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = Renderer_TensoIR_train
brdf_specular = GGX_specular


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


# ---------------------------------------------------------------------------
# Envmap loading utilities
# ---------------------------------------------------------------------------

def _recover_hdr_from_pngs(hdr_png_path, ldr_png_path=None):
    """Recover approximate linear HDR values from the encoded PNG pair.

    The preprocessing pipeline in ``preprocess_objaverse.py`` stores:

    * **HDR PNG** ``_hdr.png``:  ``uint8( log1p(10·x) / M · 255 )``
      where ``M = max( log1p(10·x) )`` over the whole image.
    * **LDR PNG** ``_ldr.png``:  ``uint8( clip(x,0,1)^(1/2.2) · 255 )``

    ``M`` is *not* stored, so we estimate it from unsaturated LDR pixels
    whose true linear value ``x = (ldr/255)^2.2`` is known.

    Returns ``np.ndarray`` of shape ``[H, W, 3]``, dtype ``float32``, in
    linear HDR space.
    """
    import cv2

    hdr_img = cv2.imread(str(hdr_png_path), cv2.IMREAD_UNCHANGED)
    if hdr_img is None:
        raise FileNotFoundError(f'Cannot read {hdr_png_path}')
    hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if ldr_png_path is not None and os.path.exists(str(ldr_png_path)):
        ldr_img = cv2.imread(str(ldr_png_path), cv2.IMREAD_UNCHANGED)
        if ldr_img is not None:
            ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x_linear = np.power(ldr_img, 2.2)  # inverse gamma → true linear [0,1]

            # For unsaturated pixels (x < ~0.95) both representations are reliable.
            # h = log1p(10·x) / M  ⟹  M = log1p(10·x) / h
            mask = (x_linear > 0.02) & (x_linear < 0.95) & (hdr_img > 1.0 / 255.0)
            if mask.sum() > 100:
                log_vals = np.log1p(10.0 * x_linear[mask])
                M_estimates = log_vals / hdr_img[mask]
                M = float(np.median(M_estimates))
                M = max(M, 0.1)
            else:
                M = np.log1p(10.0 * 5.0)  # fallback: assume max raw ≈ 5
        else:
            M = np.log1p(10.0 * 5.0)
    else:
        M = np.log1p(10.0 * 5.0)

    recovered = np.expm1(hdr_img * M) / 10.0
    recovered = np.clip(recovered, 0.0, None)
    return recovered.astype(np.float32)


def load_envmap_from_png(envmap_dir, frame_idx=0):
    """Load equirectangular envmap from the polyhaven_lvsm envmaps directory.

    Priority:
      1. Actual ``.hdr`` / ``.exr`` files (native HDR)
      2. ``_hdr.png`` + ``_ldr.png`` pair → approximate HDR recovery
      3. ``_hdr.png`` alone → approximate HDR recovery with heuristic M

    Returns ``(envmap_rgb, H, W)`` where *envmap_rgb* is a ``[H, W, 3]``
    float tensor on *device* in **linear HDR** space.
    """
    envmap_dir = Path(envmap_dir)

    for ext in ['.hdr', '.exr']:
        candidates = list(envmap_dir.glob(f'*{ext}'))
        if candidates:
            from models.relight_utils import read_hdr
            img = read_hdr(str(candidates[0]))
            img = torch.from_numpy(img).float().to(device)
            return img, img.shape[0], img.shape[1]

    hdr_png = envmap_dir / f'{frame_idx:05d}_hdr.png'
    ldr_png = envmap_dir / f'{frame_idx:05d}_ldr.png'
    if not hdr_png.exists():
        hdr_pngs = sorted(envmap_dir.glob('*_hdr.png'))
        if not hdr_pngs:
            raise FileNotFoundError(f'No envmap found in {envmap_dir}')
        hdr_png = hdr_pngs[0]
        ldr_png = Path(str(hdr_png).replace('_hdr.png', '_ldr.png'))

    img = _recover_hdr_from_pngs(hdr_png, ldr_png)
    img = torch.from_numpy(img).float().to(device)
    return img, img.shape[0], img.shape[1]


class SimpleEnvLight:
    """Lightweight environment-light class that works with equirectangular PNG
    maps (LDR or HDR) instead of requiring ``.hdr`` files.
    """

    def __init__(self, envmap_rgb, env_h, env_w):
        """
        Args:
            envmap_rgb: [H, W, 3] float tensor on *device*
        """
        self.envmap = envmap_rgb  # [H, W, 3]
        self.env_h = env_h
        self.env_w = env_w

        light_intensity = envmap_rgb.sum(dim=2, keepdim=True)  # [H, W, 1]
        h_interval = 1.0 / env_h
        sin_theta = torch.sin(
            torch.linspace(0 + 0.5 * h_interval, np.pi - 0.5 * h_interval, env_h)
        ).to(device)

        pdf = light_intensity * sin_theta.view(-1, 1, 1)
        pdf = pdf / pdf.sum()
        pdf_return = pdf * env_h * env_w / (2 * np.pi ** 2 * sin_theta.view(-1, 1, 1))

        self.pdf_sample = pdf  # [H, W, 1]
        self.pdf_return = pdf_return

        lat_step = np.pi / env_h
        lng_step = 2 * np.pi / env_w
        phi, theta = torch.meshgrid(
            torch.linspace(np.pi / 2 - 0.5 * lat_step, -np.pi / 2 + 0.5 * lat_step, env_h),
            torch.linspace(np.pi - 0.5 * lng_step, -np.pi + 0.5 * lng_step, env_w),
            indexing='ij',
        )
        view_dirs = torch.stack(
            [torch.cos(theta) * torch.cos(phi),
             torch.sin(theta) * torch.cos(phi),
             torch.sin(phi)],
            dim=-1,
        ).to(device)  # [H, W, 3]
        self.view_dirs = view_dirs

    @torch.no_grad()
    def sample_light(self, bs, num_samples):
        pdf_flat = self.pdf_sample.view(-1).expand(bs, -1)
        pdf_ret_flat = self.pdf_return.view(-1).expand(bs, -1)
        dirs_flat = self.view_dirs.view(-1, 3).expand(bs, -1, -1)
        rgb_flat = self.envmap.view(-1, 3).expand(bs, -1, -1)

        idx = torch.multinomial(pdf_flat, num_samples, replacement=True)
        light_dir = dirs_flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        light_rgb = rgb_flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        light_pdf = pdf_ret_flat.gather(1, idx).unsqueeze(-1)
        return light_dir, light_rgb, light_pdf

    @torch.no_grad()
    def get_light(self, rays_d):
        """Look up envmap colour for given ray directions [N, 3]."""
        d = torch.nn.functional.normalize(rays_d, dim=-1)
        phi = torch.asin(d[:, 2].clamp(-1, 1))
        theta = torch.atan2(d[:, 1], d[:, 0])

        u = 0.5 - theta / (2 * np.pi)
        v = 0.5 - phi / np.pi
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).unsqueeze(0).unsqueeze(0)
        envmap_chw = self.envmap.permute(2, 0, 1).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            envmap_chw, grid, align_corners=True, mode='bilinear', padding_mode='border'
        )
        return sampled.squeeze().permute(1, 0)  # [N, 3]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_scene(args, scene_name, data_root):
    """Train TensoIR on a single scene. Returns (tensoIR, logfolder)."""
    DatasetClass = dataset_dict[args.dataset_name]

    train_dataset = DatasetClass(
        data_root,
        split='train',
        downsample=args.downsample_train,
        light_name=args.light_name,
        light_rotation=args.light_rotation,
        scene_bbox=args.scene_bbox,
        scene_name=scene_name,
    )
    test_dataset = DatasetClass(
        data_root,
        split='test',
        downsample=args.downsample_test,
        light_name=args.light_name,
        light_rotation=args.light_rotation,
        scene_bbox=args.scene_bbox,
        scene_name=scene_name,
    )

    print(f'[train_scene] Loaded scene {scene_name}  '
          f'(train={len(train_dataset.all_rays)} rays, test={len(test_dataset)} views)')

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f'{args.basedir}/{scene_name}'
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/checkpoints', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    ckpt_path = f'{logfolder}/{scene_name}.th'
    ckpt_to_load = None
    min_resume_iter = 1000

    if not getattr(args, 'force_retrain', False):
        if os.path.exists(ckpt_path):
            ckpt_to_load = ckpt_path
        else:
            ckpt_dir = f'{logfolder}/checkpoints'
            if os.path.isdir(ckpt_dir):
                import re
                ckpt_files = []
                for f in os.listdir(ckpt_dir):
                    if not f.endswith('.th'):
                        continue
                    m = re.search(r'_(\d+)\.th$', f)
                    if m:
                        it = int(m.group(1))
                        if it >= min_resume_iter:
                            ckpt_files.append((it, f))
                if ckpt_files:
                    ckpt_files.sort(key=lambda x: x[0])
                    best_iter, best_file = ckpt_files[-1]
                    ckpt_to_load = os.path.join(ckpt_dir, best_file)
                    print(f'[train_scene] Found intermediate checkpoint at iter {best_iter}')

    if ckpt_to_load is not None:
        print(f'[train_scene] Loading checkpoint: {ckpt_to_load}')
        ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensoIR = eval(args.model_name)(**kwargs)
        tensoIR.load(ckpt)
        if ckpt_to_load != ckpt_path:
            tensoIR.save(ckpt_path)
            print(f'[train_scene] Promoted intermediate checkpoint → {ckpt_path}')
        return tensoIR, logfolder, test_dataset

    tensoIR = eval(args.model_name)(
        aabb, reso_cur, device,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        shadingMode=args.shadingMode,
        alphaMask_thres=args.alpha_mask_thre,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
        normals_kind=args.normals_kind,
        light_rotation=args.light_rotation,
        light_kind=args.light_kind,
        dataset=train_dataset,
        numLgtSGs=args.numLgtSGs,
    )

    grad_vars = tensoIR.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    N_voxel_list = (torch.round(torch.exp(
        torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final),
                        len(upsamp_list) + 1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs_rgb, PSNRs_rgb_brdf = [], []

    L1_reg_weight = args.L1_weight_inital
    Ortho_reg_weight = args.Ortho_weight
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()

    all_rays = train_dataset.all_rays
    all_rgbs = train_dataset.all_rgbs
    all_light_idx = train_dataset.all_light_idx

    rays_filtered, filter_mask = tensoIR.filtering_rays(all_rays, bbox_only=True)
    rgbs_filtered = all_rgbs[filter_mask, :]
    light_idx_filtered = all_light_idx[filter_mask, :]
    trainingSampler = SimpleSampler(rays_filtered.shape[0], args.batch_size)

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    relight_flag = False

    for iteration in pbar:
        rays_idx = trainingSampler.nextids()
        rays_train = rays_filtered[rays_idx]
        rgb_train = rgbs_filtered[rays_idx].to(device)
        light_idx_train = light_idx_filtered[rays_idx].to(device)
        rgb_with_brdf_train = rgb_train

        ret_kw = renderer(
            rays=rays_train,
            normal_gt=None,
            light_idx=light_idx_train,
            tensoIR=tensoIR,
            N_samples=nSamples,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            sample_method=args.light_sample_train,
            chunk_size=args.relight_chunk_size,
            is_train=True,
            is_relight=relight_flag,
            args=args,
        )

        total_loss = 0
        loss_rgb_brdf = torch.tensor(1e-6).to(device)
        loss_rgb = torch.mean((ret_kw['rgb_map'] - rgb_train) ** 2)
        total_loss += loss_rgb

        if Ortho_reg_weight > 0:
            total_loss += Ortho_reg_weight * tensoIR.vector_comp_diffs()
        if L1_reg_weight > 0:
            total_loss += L1_reg_weight * tensoIR.density_L1()
        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            total_loss += tensoIR.TV_loss_density(tvreg) * TV_weight_density
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            total_loss += tensoIR.TV_loss_app(tvreg) * TV_weight_app

        if relight_flag:
            loss_rgb_brdf = torch.mean((ret_kw['rgb_with_brdf_map'] - rgb_with_brdf_train) ** 2)
            total_loss += loss_rgb_brdf * args.rgb_brdf_weight
            nw = args.normals_loss_enhance_ratio ** (
                (iteration - update_AlphaMask_list[0]) / (args.n_iters - update_AlphaMask_list[0])
            )
            bw = args.BRDF_loss_enhance_ratio ** (
                (iteration - update_AlphaMask_list[0]) / (args.n_iters - update_AlphaMask_list[0])
            )
            if args.normals_diff_weight > 0:
                total_loss += nw * args.normals_diff_weight * ret_kw['normals_diff_map'].mean()
            if args.normals_orientation_weight > 0:
                total_loss += nw * args.normals_orientation_weight * ret_kw['normals_orientation_loss_map'].mean()
            if args.roughness_smoothness_loss_weight > 0:
                total_loss += bw * args.roughness_smoothness_loss_weight * ret_kw['roughness_smoothness_loss']
            if args.albedo_smoothness_loss_weight > 0:
                total_loss += bw * args.albedo_smoothness_loss_weight * ret_kw['albedo_smoothness_loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_rgb_val = loss_rgb.detach().item()
        PSNRs_rgb.append(-10.0 * np.log(loss_rgb_val) / np.log(10.0))
        if relight_flag:
            PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf.detach().item()) / np.log(10.0))
        else:
            PSNRs_rgb_brdf.append(0.0)

        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'{scene_name} iter {iteration:05d} '
                f'PSNR_rgb={np.mean(PSNRs_rgb[-100:]):.2f} '
                f'PSNR_brdf={np.mean(PSNRs_rgb_brdf[-100:]):.2f}'
            )

        # Periodic test-set visualization
        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0 and relight_flag:
            from renderer import evaluation_iter_TensoIR_simple
            PSNRs_test, PSNRs_rgb_brdf_test = evaluation_iter_TensoIR_simple(
                test_dataset, tensoIR, args, renderer,
                f'{logfolder}/imgs_vis/',
                prtx=f'{iteration:06d}_',
                N_samples=nSamples,
                white_bg=white_bg, ndc_ray=ndc_ray,
                compute_extra_metrics=False,
                logger=summary_writer, step=iteration, device=device,
            )
            summary_writer.add_scalar('test/psnr_rgb', np.mean(PSNRs_test), global_step=iteration)
            summary_writer.add_scalar('test/psnr_rgb_brdf', np.mean(PSNRs_rgb_brdf_test), global_step=iteration)

        # Periodic checkpoint save (skip iteration 0)
        if iteration > 0 and iteration % args.save_iters == 0:
            tensoIR.save(f'{logfolder}/checkpoints/{scene_name}_{iteration}.th')

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256 ** 3:
                reso_mask = reso_cur
            new_aabb = tensoIR.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensoIR.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                relight_flag = True
                torch.cuda.empty_cache()
                TV_weight_density = 0
                TV_weight_app = 0
            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                rays_filtered, filter_mask = tensoIR.filtering_rays(all_rays, bbox_only=True)
                rgbs_filtered = all_rgbs[filter_mask, :]
                light_idx_filtered = all_light_idx[filter_mask, :]
                trainingSampler = SimpleSampler(rays_filtered.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensoIR.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensoIR.upsample_volume_grid(reso_cur)
            if args.lr_upsample_reset:
                lr_scale = 1
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensoIR.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    tensoIR.save(ckpt_path)
    print(f'[train_scene] Saved checkpoint to {ckpt_path}')
    return tensoIR, logfolder, test_dataset


# ---------------------------------------------------------------------------
# Relighting
# ---------------------------------------------------------------------------

@torch.no_grad()
def relight_scene(tensoIR, dataset, envir_light, out_dir,
                  target_indices=None, acc_thre=0.5,
                  vis_equation='nerv', batch_size=4096):
    """Relight *dataset* views under *envir_light* and save to *out_dir*."""
    W, H = dataset.img_wh
    os.makedirs(out_dir, exist_ok=True)

    if target_indices is None:
        target_indices = list(range(len(dataset)))

    for view_idx in tqdm(target_indices, desc='Relighting'):
        item = dataset[view_idx]
        frame_rays = item['rays'].to(device)         # [H*W, 6]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device)

        rgb_map, depth_map, normal_map, albedo_map = [], [], [], []
        roughness_map, fresnel_map, acc_map = [], [], []
        relight_rgb_chunks = []

        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), batch_size)
        for chunk_idx in chunk_idxs:
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *_ = tensoIR(
                        frame_rays[chunk_idx], light_idx[chunk_idx],
                        is_train=False, white_bg=True, ndc_ray=False, N_samples=-1,
                    )

            acc_mask = (acc_chunk > acc_thre)
            rays_o_chunk = frame_rays[chunk_idx][:, :3]
            rays_d_chunk = frame_rays[chunk_idx][:, 3:]
            surface_xyz = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk

            relight_rgb = torch.ones_like(rgb_chunk)  # white bg

            if acc_mask.any():
                masked_pts = surface_xyz[acc_mask]
                masked_normal = normal_chunk[acc_mask]
                masked_albedo = albedo_chunk[acc_mask]
                masked_roughness = roughness_chunk[acc_mask]
                masked_fresnel = fresnel_chunk[acc_mask]

                surf2c = safe_l2_normalize(-rays_d_chunk[acc_mask], dim=-1)
                light_dir, light_rgb, light_pdf = envir_light.sample_light(masked_pts.shape[0], 512)

                cosine = torch.einsum('ijk,ik->ij', light_dir, masked_normal)
                cosine_mask = (cosine > 1e-6)

                visibility = torch.zeros(*cosine_mask.shape, 1, device=device)
                masked_xyz_exp = masked_pts[:, None, :].expand(*cosine_mask.shape, 3)

                cos_pts = masked_xyz_exp[cosine_mask]
                cos_l = light_dir[cosine_mask]
                cos_vis = torch.zeros(cos_l.shape[0], 1, device=device)

                vis_chunks = torch.split(torch.arange(cos_pts.shape[0]), 100000)
                for vc in vis_chunks:
                    nerv_vis, nerfactor_vis = compute_transmittance(
                        tensoIR=tensoIR,
                        surf_pts=cos_pts[vc],
                        light_in_dir=cos_l[vc],
                        nSample=96, vis_near=0.05, vis_far=1.5,
                    )
                    if vis_equation == 'nerfactor':
                        cos_vis[vc] = nerfactor_vis.unsqueeze(-1)
                    else:
                        cos_vis[vc] = nerv_vis.unsqueeze(-1)
                visibility[cosine_mask] = cos_vis

                nlights = light_dir.shape[1]
                specular = brdf_specular(masked_normal, surf2c, light_dir, masked_roughness, masked_fresnel)
                surface_brdf = masked_albedo.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular
                direct = visibility * light_rgb
                contrib = surface_brdf * direct * cosine[:, :, None] / light_pdf
                surface_rgb = contrib.mean(dim=1)
                surface_rgb = torch.clamp(surface_rgb, 0.0, 1.0)
                if surface_rgb.shape[0] > 0:
                    surface_rgb = linear2srgb_torch(surface_rgb)
                relight_rgb[acc_mask] = surface_rgb

            relight_rgb_chunks.append(relight_rgb.cpu())
            acc_map.append(acc_chunk.cpu())

        relight_img = torch.cat(relight_rgb_chunks, 0).reshape(H, W, 3).numpy()
        acc_full = torch.cat(acc_map, 0)

        bg_color = envir_light.get_light(frame_rays[:, 3:])
        bg_color = torch.clamp(bg_color, 0.0, 1.0)
        bg_color = linear2srgb_torch(bg_color).cpu().reshape(H, W, 3).numpy()

        acc_np = acc_full.reshape(H, W, 1).numpy()
        acc_np[acc_np <= 0.9] = 0.0
        relight_with_bg = acc_np * relight_img + (1.0 - acc_np) * bg_color

        fname = f'{view_idx:05d}.png'
        imageio.imwrite(os.path.join(out_dir, fname), (relight_with_bg * 255).astype('uint8'))
        imageio.imwrite(os.path.join(out_dir, f'{view_idx:05d}_nobg.png'), (relight_img * 255).astype('uint8'))

    print(f'[relight_scene] Saved {len(target_indices)} relit views to {out_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Batch TensoIR train + relight for polyhaven_lvsm')
    parser.add_argument('--config', type=str, required=True, help='Path to TensoIR config txt')
    parser.add_argument('--data_root', type=str, required=True, help='Root of polyhaven_lvsm dataset (contains metadata/, images/, envmaps/)')
    parser.add_argument('--relight_meta_dir', type=str, required=True, help='Directory with relight_metadata JSON files')
    parser.add_argument('--output_dir', type=str, default='./output/polyhaven_relight', help='Root output directory')
    parser.add_argument('--force_retrain', action='store_true', help='Force retrain even if checkpoint exists')
    parser.add_argument('--relight_only', action='store_true', help='Skip training, only do relighting (requires existing ckpt)')
    parser.add_argument('--vis_equation', type=str, default='nerv', choices=['nerv', 'nerfactor'])
    parser.add_argument('--acc_thre', type=float, default=0.5)
    parser.add_argument('--relight_batch_size', type=int, default=4096)
    batch_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ['--config', batch_args.config] + remaining
    args = config_parser()
    args.dataset_name = 'polyhaven_lvsm'
    args.datadir = batch_args.data_root
    args.basedir = os.path.join(batch_args.output_dir, 'training')
    args.force_retrain = batch_args.force_retrain

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)

    meta_dir = Path(batch_args.relight_meta_dir)
    meta_files = sorted(meta_dir.glob('*.json'))
    print(f'Found {len(meta_files)} relight metadata files')

    trained_scenes = {}

    for meta_file in meta_files:
        with open(meta_file) as f:
            meta = json.load(f)

        scene_name = meta['scene_name']
        relit_scene_name = meta['relit_scene_name']
        target_indices = meta.get('target_view_indices', None)

        print(f'\n{"="*60}')
        print(f'Processing: {meta_file.stem}')
        print(f'  scene_name     = {scene_name}')
        print(f'  relit_scene    = {relit_scene_name}')
        print(f'  target_indices = {target_indices}')
        print(f'{"="*60}')

        # --- Train ---
        if scene_name not in trained_scenes and not batch_args.relight_only:
            print(f'\n>>> Training scene: {scene_name}')
            tensoIR, logfolder, test_dataset = train_scene(args, scene_name, batch_args.data_root)
            trained_scenes[scene_name] = (tensoIR, logfolder, test_dataset)
        elif scene_name in trained_scenes:
            tensoIR, logfolder, test_dataset = trained_scenes[scene_name]
        else:
            ckpt_path = f'{args.basedir}/{scene_name}/{scene_name}.th'
            if not os.path.exists(ckpt_path):
                print(f'[SKIP] No checkpoint for {scene_name} and --relight_only is set')
                continue
            print(f'Loading checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            kwargs = ckpt['kwargs']
            kwargs.update({'device': device})
            tensoIR = eval(args.model_name)(**kwargs)
            tensoIR.load(ckpt)

            DatasetClass = dataset_dict[args.dataset_name]
            test_dataset = DatasetClass(
                batch_args.data_root,
                split='test',
                downsample=args.downsample_test,
                scene_name=scene_name,
            )
            logfolder = f'{args.basedir}/{scene_name}'
            trained_scenes[scene_name] = (tensoIR, logfolder, test_dataset)

        # --- Relight ---
        envmap_dir = Path(batch_args.data_root) / 'envmaps' / relit_scene_name
        if not envmap_dir.exists():
            print(f'[WARN] Envmap dir not found: {envmap_dir}, skipping relighting for {meta_file.stem}')
            continue

        print(f'\n>>> Relighting {scene_name} with envmap from {relit_scene_name}')
        envmap_rgb, env_h, env_w = load_envmap_from_png(envmap_dir)
        envir_light = SimpleEnvLight(envmap_rgb, env_h, env_w)

        relight_out = os.path.join(batch_args.output_dir, 'relight', meta_file.stem)
        relight_scene(
            tensoIR, test_dataset, envir_light, relight_out,
            target_indices=target_indices,
            acc_thre=batch_args.acc_thre,
            vis_equation=batch_args.vis_equation,
            batch_size=batch_args.relight_batch_size,
        )

    print('\n' + '=' * 60)
    print('All done!')
    print(f'Training logs:  {args.basedir}')
    print(f'Relight output: {os.path.join(batch_args.output_dir, "relight")}')
    print('=' * 60)


if __name__ == '__main__':
    main()
