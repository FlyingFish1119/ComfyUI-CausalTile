from .llm_client import load_cached_models, parse_headers, resolve_api_key

LLM_TYPE = "CAUSALTILE_LLM"
AUTO_DISCOVER = "（点刷新获取模型列表）"
HEADERS_HINT = "一行一个 key: value"


class LLMModelLoader:

    DESCRIPTION = "配置一个 OpenAI 兼容的 LLM 接口。api_key 支持 env:变量名；headers 一行一个 key: value。"

    @classmethod
    def INPUT_TYPES(cls):
        models = load_cached_models() or [AUTO_DISCOVER]
        return {
            "required": {
                "base_url": ("STRING", {"default": "https://open.bigmodel.cn/api/paas/v4"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (models,),
            },
            "optional": {
                "headers": ("STRING", {"multiline": True, "default": "", "placeholder": HEADERS_HINT}),
            },
        }

    RETURN_TYPES = (LLM_TYPE,)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load"
    CATEGORY = "CausalTile"

    def load(self, base_url, api_key, model, headers=""):
        return (
            {
                "base_url": (base_url or "").strip(),
                "api_key": resolve_api_key(api_key),
                "model": "" if model == AUTO_DISCOVER else (model or "").strip(),
                "headers": parse_headers(headers),
            },
        )
