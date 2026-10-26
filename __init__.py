from .core.node_detailer import CausalTileDetailer
from .core.node_prompt import CausalTilePrompt
from .core.node_split import CausalTileSplit
from .llm.llm_image_to_prompt import LLMImageToPrompt
from .llm.llm_model_loader import LLMModelLoader
from .llm.server_routes import register_routes

NODE_CLASS_MAPPINGS = {
    "CausalTileSplit": CausalTileSplit,
    "CausalTilePrompt": CausalTilePrompt,
    "CausalTileDetailer": CausalTileDetailer,
    "LLMModelLoader": LLMModelLoader,
    "LLMImageToPrompt": LLMImageToPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CausalTileSplit": "CausalTile 拆图",
    "CausalTilePrompt": "CausalTile 设置提示词",
    "CausalTileDetailer": "CausalTile 分块重绘",
    "LLMModelLoader": "LLM Model Loader",
    "LLMImageToPrompt": "LLM Image To Prompt",
}

WEB_DIRECTORY = "./web"

register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
