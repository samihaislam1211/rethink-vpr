"""
extract_latents.py

Generates synthetic views at multiple azimuth angles for a given input image using GenWarp.
For each viewpoint it:
  - Saves the full synthesized image (diffusion output)
  - Saves the geometry-warped image (before diffusion)
  - Saves the "Checkpoint A" latent: VAE(warped_image), shape (4, 64, 64)
    This is after the VAE encoder and viewpoint warp, before any diffusion denoising.

Usage:
    python extract_latents.py \\
        --image assets/pexels-heyho-5998120_19mm.jpg \\
        --focal_mm 19 \\
        --output_dir output_latents \\
        --checkpoint multi1 \\
        --angles 0 10 20 30 45 60

Optional flags:
    --ele_deg       elevation angle in degrees (default: 10)
    --radius        camera distance offset in metres (default: 0.5)
    --no_zoedepth   skip ZoeDepth, load depth from --depth_path instead
    --depth_path    path to a pre-computed depth .pt or .png file
    --seed          random seed for reproducibility
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

sys.path.insert(0, os.path.dirname(__file__))

from genwarp import GenWarp
from genwarp.ops import (
    camera_lookat, get_projection_matrix, sph2cart, focal_length_to_fov
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_crop(image_path: str, res: int = 512) -> Image.Image:
    """Centre-crop image to square then resize."""
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    if W < H:
        left, right = 0, W
        top = int(np.ceil((H - W) / 2.0))
        bottom = top + W
    else:
        top, bottom = 0, H
        left = int(np.ceil((W - H) / 2.0))
        right = left + H
    img = img.crop((left, top, right, bottom))
    return img.resize((res, res), Image.LANCZOS)


def make_camera_matrices(
    azi_deg: float,
    ele_deg: float,
    radius: float,
    mean_depth: float,
    fovy: float,
    near: float = 0.01,
    far: float = 100.0,
    dtype=torch.float16,
    device: str = 'cuda',
):
    """Return (src_proj_mtx, tar_proj_mtx, rel_view_mtx) for a given viewpoint."""
    fovy_t = torch.ones(1) * fovy

    proj_mtx = get_projection_matrix(
        fovy=fovy_t, aspect_wh=1.0, near=near, far=far
    ).to(device, dtype=dtype)

    z_up = torch.tensor([[0., 0., 1.]])

    # Source camera: ZoeDepth reference (looking toward -x from origin)
    src_view_mtx = camera_lookat(
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[-1., 0., 0.]]),
        z_up,
    )

    # Target camera
    # sph2cart expects r to be a 1-D tensor of shape (B,); use [val] not val
    azi = torch.tensor(np.deg2rad(azi_deg))
    ele = torch.tensor(np.deg2rad(ele_deg))
    r   = torch.tensor([mean_depth + radius])          # (1,)
    eye = sph2cart(azi, ele, r).float()                # (1, 3)
    at  = F.pad(
        torch.tensor([[-mean_depth]]), (0, 2), mode='constant', value=0.0
    )  # (1, 3)  centre of scene
    tar_view_mtx = camera_lookat(eye + at, at, z_up)

    rel_view_mtx = (
        tar_view_mtx @ torch.linalg.inv(src_view_mtx.float())
    ).to(device, dtype=dtype)

    return proj_mtx, proj_mtx, rel_view_mtx   # src_proj == tar_proj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to source image')
    parser.add_argument('--focal_mm', type=float, default=None,
                        help='35mm-equivalent focal length in mm')
    parser.add_argument('--fov_deg', type=float, default=None,
                        help='Vertical FOV in degrees (alternative to focal_mm)')
    parser.add_argument('--output_dir', default='output_latents')
    parser.add_argument('--checkpoint', default='multi1',
                        help='Checkpoint folder name inside checkpoints/')
    parser.add_argument('--angles', nargs='+', type=float,
                        default=[5, 10, 20, 30, 45, 60],
                        help='Elevation angles in degrees (the sweep axis)')
    parser.add_argument('--azi_deg', type=float, default=0.0,
                        help='Fixed azimuth angle in degrees (default: 0)')
    parser.add_argument('--radius', type=float, default=0.5,
                        help='Camera distance offset in metres')
    parser.add_argument('--res', type=int, default=512,
                        help='Image resolution (square)')
    parser.add_argument('--no_zoedepth', action='store_true')
    parser.add_argument('--depth_path', default=None,
                        help='Path to pre-computed depth (.pt tensor or grayscale image)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Validate FOV specification
    if args.focal_mm is None and args.fov_deg is None:
        parser.error('Provide --focal_mm or --fov_deg')

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directories
    out_root = Path(args.output_dir)
    latent_dir = out_root / 'latents'
    image_dir  = out_root / 'images'
    latent_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load GenWarp
    # -----------------------------------------------------------------------
    genwarp_cfg = dict(
        pretrained_model_path='checkpoints',
        checkpoint_name=args.checkpoint,
        half_precision_weights=True,
    )
    genwarp_nvs = GenWarp(cfg=genwarp_cfg, device=device)

    # -----------------------------------------------------------------------
    # Load and preprocess source image
    # -----------------------------------------------------------------------
    pil_img = load_and_crop(args.image, args.res)
    # Load as float32 first — ZoeDepth requires float32 input.
    src_image_f32 = to_tensor(pil_img)[None].to(device, dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Estimate or load depth
    # -----------------------------------------------------------------------
    if args.no_zoedepth:
        if args.depth_path is None:
            raise ValueError('--depth_path required when --no_zoedepth is set')
        dp = args.depth_path
        if dp.endswith('.pt'):
            src_depth = torch.load(dp).to(device, dtype=torch.float16)
        else:
            depth_pil = Image.open(dp).convert('L').resize((args.res, args.res))
            src_depth = to_tensor(depth_pil)[None].to(device, dtype=torch.float16)
    else:
        print('Loading ZoeDepth...')
        zoedepth = torch.hub.load(
            'extern/ZoeDepth',
            'ZoeD_N',
            source='local',
            pretrained=True,
            trust_repo=True,
        ).to(device)
        print('Estimating depth...')
        with torch.no_grad():
            # ZoeDepth must receive float32; cast depth output to float16 afterwards.
            src_depth = zoedepth.infer(src_image_f32).half()

    # Convert image to float16 after depth estimation (matches notebook convention).
    src_image = src_image_f32.half()

    # -----------------------------------------------------------------------
    # FOV
    # -----------------------------------------------------------------------
    if args.focal_mm is not None:
        fovy = focal_length_to_fov(args.focal_mm, 24.0)
    else:
        fovy = float(np.deg2rad(args.fov_deg))

    mean_depth = float(src_depth.mean(dim=(2, 3)).squeeze().cpu())
    print(f'Mean scene depth: {mean_depth:.3f} m')

    # Save source image and depth for reference
    pil_img.save(image_dir / 'source.png')
    depth_norm = (src_depth[0, 0].float().cpu().numpy())
    depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
    Image.fromarray((depth_norm * 255).astype(np.uint8)).save(image_dir / 'depth.png')

    # -----------------------------------------------------------------------
    # Per-angle inference
    # -----------------------------------------------------------------------
    metadata = {
        'source_image': args.image,
        'focal_mm': args.focal_mm,
        'fov_deg': args.fov_deg,
        'azi_deg': args.azi_deg,
        'radius': args.radius,
        'mean_depth': mean_depth,
        'seed': args.seed,
        'checkpoint': args.checkpoint,
        'angles': [],
    }

    for ele_deg in args.angles:
        print(f'\n--- Elevation {ele_deg:+.1f}° ---')

        src_proj_mtx, tar_proj_mtx, rel_view_mtx = make_camera_matrices(
            azi_deg=args.azi_deg,
            ele_deg=ele_deg,
            radius=args.radius,
            mean_depth=mean_depth,
            fovy=fovy,
            device=device,
            dtype=torch.float16,
        )

        # Full inference
        with torch.no_grad():
            renders = genwarp_nvs(
                src_image=src_image,
                src_depth=src_depth,
                rel_view_mtx=rel_view_mtx,
                src_proj_mtx=src_proj_mtx,
                tar_proj_mtx=tar_proj_mtx,
            )

        synthesized = renders['synthesized']   # (1, 3, H, W) in [0,1]
        warped      = renders['warped']         # (1, 3, H, W) in [0,1]

        # ------------------------------------------------------------------
        # Checkpoint A: VAE(warped_image)
        #   This is the latent representation AFTER the viewpoint warp and
        #   AFTER the VAE encoder, but BEFORE any diffusion denoising.
        #   Shape: (1, 4, 64, 64)
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Pass warped in the VAE's native dtype (float16 when half_precision_weights=True).
            # Calling .float() would cause a dtype mismatch with the half-precision VAE weights.
            checkpoint_a = genwarp_nvs.encode_images(warped.to(dtype=genwarp_nvs.dtype))

        # Convert to float32 for saving (avoids fp16 precision issues in .pt)
        checkpoint_a_f32 = checkpoint_a.squeeze(0).float().cpu()   # (4, 64, 64)

        # ------------------------------------------------------------------
        # Save artefacts
        # ------------------------------------------------------------------
        tag = f'ele_{ele_deg:06.1f}'

        latent_path = latent_dir / f'{tag}.pt'
        torch.save(checkpoint_a_f32, latent_path)

        syn_path  = image_dir / f'{tag}_synthesized.png'
        warp_path = image_dir / f'{tag}_warped.png'
        to_pil_image(synthesized[0].float().cpu()).save(syn_path)
        to_pil_image(warped[0].float().cpu()).save(warp_path)

        print(f'  Latent saved  : {latent_path}  shape={tuple(checkpoint_a_f32.shape)}')
        print(f'  Synthesized   : {syn_path}')
        print(f'  Warped        : {warp_path}')

        metadata['angles'].append({
            'ele_deg': ele_deg,
            'latent_file': str(latent_path.relative_to(out_root)),
            'synthesized_file': str(syn_path.relative_to(out_root)),
            'warped_file': str(warp_path.relative_to(out_root)),
            'latent_shape': list(checkpoint_a_f32.shape),
            'latent_mean': float(checkpoint_a_f32.mean()),
            'latent_std': float(checkpoint_a_f32.std()),
        })

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------
    meta_path = out_root / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nMetadata saved to {meta_path}')
    print(f'\nDone. All outputs in: {out_root}')


if __name__ == '__main__':
    main()
