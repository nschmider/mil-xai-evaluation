import math
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide
from PIL import Image
import torch
from torchmil.datasets import CAMELYON16MILDataset


def compute_slides(patch_scores, path, alpha=0.5, gt=None):
    t1 = time.time()
    patch_size = 512
    for i, attribution_scores in enumerate(patch_scores):
        if i == 7:
            return
        print(i)
        img_num = i + 1 + (i >= 48)
        with OpenSlide(
            f"data/test_slides/CAMELYON16/testing/images/test_{img_num:03d}.tif"
        ) as img:
            train_dataset = CAMELYON16MILDataset(
                root="data", features="UNI", partition="test"
            )
            coords = torch.tensor(
                np.load(f"data/patches_512/coords/test_{img_num:03d}.npy")
            )
            coords = coords // 512

            heatmap = torch.zeros(
                (
                    math.ceil(img.dimensions[1] / patch_size),
                    math.ceil(img.dimensions[0] / patch_size),
                )
            )

            if gt is None:
                attribution_scores = patch_scores[i]
                max_abs = attribution_scores.abs().max()
                scores = attribution_scores / max_abs
                for coord, score in zip(coords, scores):
                    x = int(coord[0])
                    y = int(coord[1])
                    heatmap[y, x] = score
                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                cmap = plt.get_cmap("RdYlBu_r")
                heatmap_np = heatmap.numpy()
                colors = cmap(norm(heatmap_np))
                colors = (colors * 255).astype(np.uint8)
                colors[:, :, 3] = np.where(heatmap_np != 0, alpha * 255, 0)
            else:
                for coord, label in zip(coords, gt[i]):
                    x = int(coord[0])
                    y = int(coord[1])
                    heatmap[y, x] = label
                colors = np.zeros((*heatmap.shape, 4), dtype=np.uint8)
                colors[heatmap == 0] = [0, 0, 0, 0]
                colors[heatmap == 1] = [255, 0, 0, int(alpha * 255)]
            overlay = Image.fromarray(colors, "RGBA")
            thumbnail = img.read_region((0, 0), 8, img.level_dimensions[8]).convert(
                "RGBA"
            )
            overlay = overlay.resize(thumbnail.size, Image.Resampling.NEAREST)
            result = Image.alpha_composite(thumbnail, overlay)
            result.save(f"{path}/test_slide_{i}.png")
    t2 = time.time()
    return t2 - t1
