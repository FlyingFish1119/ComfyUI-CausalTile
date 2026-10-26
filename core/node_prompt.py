from __future__ import annotations

import logging

from .causal_core import apply_prompts, first_or_single

LOGGER = logging.getLogger(__name__)


class CausalTilePrompt:
    """将提示词列表按行优先写入 plan（传单条则所有分块复用）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("CAUSAL_TILE_PLAN",),
                "prompts": ("STRING", {"default": "", "multiline": False}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("CAUSAL_TILE_PLAN",)
    RETURN_NAMES = ("plan",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "apply"
    CATEGORY = "CausalTile"

    def apply(self, plan, prompts):
        plan = first_or_single(plan)
        prompts = prompts if isinstance(prompts, list) else [prompts]

        new_plan = apply_prompts(plan, prompts)
        LOGGER.info(
            "CausalTilePrompt: 已写入 %d 条提示词，共 %d 块",
            len(prompts), len(new_plan.tiles),
        )
        return (new_plan,)
