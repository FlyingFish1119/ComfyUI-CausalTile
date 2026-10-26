from __future__ import annotations

import logging

import comfy.samplers

from .causal_core import run_causal_detail

LOGGER = logging.getLogger(__name__)


class CausalTileDetailer:
    """按 plan 顺序逐块裁剪视野框、原尺寸重绘并单向羽化贴回 latent 画布。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "tile_plan": ("CAUSAL_TILE_PLAN",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "first_denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "edge_denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4}),
                "noise_mask_feather": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "detail"
    CATEGORY = "CausalTile"

    def detail(self,
               latent,
               tile_plan,
               model,
               clip,
               negative_prompt,
               seed,
               steps,
               cfg,
               sampler_name,
               scheduler,
               denoise,
               first_denoise,
               edge_denoise,
               feather,
               noise_mask_feather):
        # 这一行打出来，说明 ComfyUI 已经把 model/clip 搬进显存、真的进了节点函数
        # 看不到这一行，就是卡在 ComfyUI 自己的模型加载阶段，跟本节点的逻辑无关
        print("[CausalTile] 节点已进入执行：模型与 LATENT 就绪", flush=True)
        plan = tile_plan
        samples = latent["samples"]

        final = run_causal_detail(
            latent_canvas=samples,
            plan=plan,
            model=model,
            clip=clip,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            first_denoise=first_denoise,
            edge_denoise=edge_denoise,
            feather=feather,
            noise_mask_feather=noise_mask_feather,
        )

        LOGGER.info(
            "CausalTileDetailer: 完成 %d 块, 输出 latent %s（对应 %dx%d 像素），解码交给下游节点",
            len(plan.tiles), tuple(final.shape), plan.width, plan.height,
        )
        return ({"samples": final},)
