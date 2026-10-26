from __future__ import annotations

import logging

from .causal_core import pad_image, plan_tiles, vae_compression

LOGGER = logging.getLogger(__name__)


class CausalTileSplit:
    """将画布按行优先切分为基础块并生成几何规划。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "rows": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "overlap_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "view_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CAUSAL_TILE_PLAN", "IMAGE")
    RETURN_NAMES = ("tile_plan", "tile_images")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "split"
    CATEGORY = "CausalTile"

    def split(self, image, vae, rows, columns, overlap_ratio, view_factor):
        height, width = int(image.shape[1]), int(image.shape[2])
        # 压缩倍数向 vae 要，几何对齐、overlap 取整、latent 坐标全部跟着它走
        align = vae_compression(vae)
        plan = plan_tiles(
            width=width,
            height=height,
            rows=int(rows),
            columns=int(columns),
            overlap_ratio=float(overlap_ratio),
            view_factor=float(view_factor),
            align=align,
        )

        canvas = pad_image(image, plan.pad_width, plan.pad_height)
        # 输出视野框（view_box）而不是基础块（tile_box）：
        # 视野框把左边、上边的邻居区域也带上了，VLM 能看见这块在整张图里的上下文，
        # 反推出来的提示词就不会只管本块、跟周围脱节。
        tile_images = [
            canvas[:, ty1:ty2, tx1:tx2, :].contiguous()
            for tx1, ty1, tx2, ty2 in (t.view_box for t in plan.tiles)
        ]

        LOGGER.info(
            "CausalTileSplit: %dx%d -> pad %dx%d, %d 行 x %d 列, 共 %d 块, vae 压缩倍数 %d",
            width, height, plan.pad_width, plan.pad_height,
            plan.rows, plan.columns, len(plan.tiles), align,
        )
        return (plan, tile_images)
