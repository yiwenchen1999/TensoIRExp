import os, json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import get_ray_directions, get_rays


class PolyhavenLVSM_Dataset(Dataset):
    """
    DataLoader for the polyhaven_lvsm format. Metadata is a single JSON per scene
    containing w2c matrices (OpenCV convention) and fxfycxcy intrinsics.
    """

    def __init__(
        self,
        root_dir,
        hdr_dir=None,
        split='train',
        random_test=False,
        N_vis=-1,
        downsample=1.0,
        sub=0,
        light_rotation=None,
        light_name="env_0",
        scene_name=None,
        near_far=None,
        scene_bbox=None,
        **kwargs,
    ):
        assert split in ['train', 'test']
        self.N_vis = N_vis
        self.root_dir = Path(root_dir)
        self.split = split
        self.downsample = downsample
        self.white_bg = True
        self.transform = T.Compose([T.ToTensor()])
        self.light_name = light_name

        if light_rotation is None:
            light_rotation = ['000']
        self.light_rotation = light_rotation
        self.light_num = len(self.light_rotation)

        if scene_name is None:
            metadata_dir = self.root_dir / 'metadata'
            scene_name = sorted([f.stem for f in metadata_dir.glob('*.json')])[0]
        self.scene_name = scene_name

        meta_path = self.root_dir / 'metadata' / f'{self.scene_name}.json'
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        frames = self.meta['frames']
        if not random_test:
            pass
        if sub > 0:
            frames = frames[:sub]
        self.frames = frames

        first = frames[0]
        fxfycxcy = first['fxfycxcy']
        self.fx, self.fy = fxfycxcy[0], fxfycxcy[1]
        self.cx, self.cy = fxfycxcy[2], fxfycxcy[3]

        sample_img_path = self._resolve_image_path(first['image_path'])
        sample_img = Image.open(sample_img_path)
        orig_w, orig_h = sample_img.size

        self.img_wh = (int(orig_w / downsample), int(orig_h / downsample))
        W, H = self.img_wh

        focal_x = self.fx / downsample
        focal_y = self.fy / downsample
        cx = self.cx / downsample
        cy = self.cy / downsample

        self.directions = get_ray_directions(H, W, [focal_x, focal_y], center=[cx, cy])
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        if near_far is not None:
            self.near_far = near_far
        else:
            self.near_far = [0.5, 3.5]

        if scene_bbox is not None:
            self.scene_bbox = torch.tensor(scene_bbox).float()
        else:
            self.scene_bbox = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).float()

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        self.lights_probes = None
        self.hdr_dir = Path(hdr_dir) if hdr_dir else None
        self._read_lights()

        if split == 'train':
            self.read_all_frames()

    def _resolve_image_path(self, json_path):
        """Resolve an image path from the JSON, falling back to local images/ dir."""
        if os.path.exists(json_path):
            return json_path
        fname = os.path.basename(json_path)
        scene_dir = os.path.basename(os.path.dirname(json_path))
        local = self.root_dir / 'images' / scene_dir / fname
        if local.exists():
            return str(local)
        return json_path

    def _read_lights(self):
        if self.hdr_dir is None or not self.hdr_dir.exists():
            return
        for ext in ['.hdr', '.exr']:
            candidates = list(self.hdr_dir.glob(f'*{ext}'))
            if candidates:
                break
        if not candidates:
            return
        from models.relight_utils import read_hdr
        hdr_path = candidates[0]
        light_rgb = read_hdr(str(hdr_path))
        if light_rgb is not None:
            self.envir_map_h, self.envir_map_w = light_rgb.shape[:2]
            light_rgb = light_rgb.reshape(-1, 3)
            self.lights_probes = torch.from_numpy(light_rgb).float()

    def read_all_frames(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_light_idx = []

        for idx in tqdm(range(len(self.frames)), desc=f'Loading {self.split} data ({self.scene_name})'):
            frame = self.frames[idx]
            w2c = torch.FloatTensor(frame['w2c'])
            c2w = torch.linalg.inv(w2c)

            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)

            img_path = self._resolve_image_path(frame['image_path'])
            img = Image.open(img_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)
            C = img.shape[0]
            img = img.view(C, -1).permute(1, 0)

            if C == 4:
                rgbs = img[:, :3] * img[:, 3:4] + (1 - img[:, 3:4])
                mask = (img[:, 3:4] > 0).to(torch.bool)
            else:
                rgbs = img[:, :3]
                mask = torch.ones((rgbs.shape[0], 1), dtype=torch.bool)

            light_idx = torch.zeros((self.img_wh[0] * self.img_wh[1], 1), dtype=torch.int8)

            self.all_rays.append(rays)
            self.all_rgbs.append(rgbs)
            self.all_masks.append(mask)
            self.all_light_idx.append(light_idx)

        self.all_rays = torch.cat(self.all_rays, dim=0)
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)
        self.all_masks = torch.cat(self.all_masks, dim=0)
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        w2c = torch.FloatTensor(frame['w2c'])
        c2w = torch.linalg.inv(w2c)

        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)

        img_path = self._resolve_image_path(frame['image_path'])
        img = Image.open(img_path)
        if self.downsample != 1.0:
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
        img = self.transform(img)
        C = img.shape[0]
        img = img.view(C, -1).permute(1, 0)

        if C == 4:
            rgbs = img[:, :3] * img[:, 3:4] + (1 - img[:, 3:4])
            mask = (img[:, 3:4] > 0).to(torch.bool)
        else:
            rgbs = img[:, :3]
            mask = torch.ones((rgbs.shape[0], 1), dtype=torch.bool)

        light_idx = torch.zeros((self.img_wh[0] * self.img_wh[1], 1), dtype=torch.int)

        rgbs = rgbs.unsqueeze(0)
        light_idx = light_idx.unsqueeze(0)

        item = {
            'img_wh': self.img_wh,
            'light_idx': light_idx,
            'rgbs': rgbs,
            'rgbs_mask': mask,
            'rays': rays,
            'c2w': c2w,
            'w2c': w2c,
        }
        return item
