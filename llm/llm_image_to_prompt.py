from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .llm_client import chat_completion, fetch_models, image_to_data_url, strip_code_fence

DEFAULT_SYSTEM_PROMPT = """你是一名严格的局部图像提示词反推助手。

请结合【全局系统提示词】与【局部图像】，只分析局部图像中实际可见的内容，并生成一条适用于图像生成模型的英文提示词，用来描述和重建该区域。

全局系统提示词用于提供角色身份、固定画风、服装设定、色彩方向、构图逻辑和不可违反的限制；局部图像用于确认该区域真实可见的姿势、身体部位、服装细节、配饰、材质、动作、表情、光影、线条和局部构图。优先服从全局系统提示词，局部图像中清晰可见的信息优先于推测；如果两者冲突，保留全局设定并仅修正局部可见内容。不要描述局部图像之外的身体、背景或物体，不要根据常识补全被遮挡区域，不要臆测不可见的颜色、材质、文字、品牌、姿势或细节；无法确认的内容直接省略。不要输出分析过程、解释、引号、项目符号或负面提示词。

采用自然语言作为提示词，提示词应包含：局部区域主体、可见身体部位或物体、服装与配饰、动作与姿势、表情、材质与线条、局部光影、画风和构图；避免重复全局提示词中与该区域无关的内容，也不要擅自添加新的角色设定、体型描述、毛色描述或风格。"""

DEFAULT_USER_PROMPT = "Describe this image."


def first_or_single(value, default=None):
    """列表输入取第一个，单值原样返回。"""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def flatten_images(value):
    """把 image 输入摊成单张图像列表，兼容单张、带 batch 的 tensor 和嵌套列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        images = []
        for item in value:
            images.extend(flatten_images(item))
        return images
    if hasattr(value, "ndim") and value.ndim == 4:
        return [value[i] for i in range(value.shape[0])]
    return [value]


class LLMImageToPrompt:

    DESCRIPTION = "把图片交给 LLM 反推成提示词文本；一次收到多张图时并发请求，返回顺序与输入一致。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("CAUSALTILE_LLM",),
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "user_prompt": ("STRING", {"multiline": True, "default": DEFAULT_USER_PROMPT}),
                "max_side": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "timeout": ("INT", {"default": 120, "min": 1, "max": 86400, "step": 1}),
            },
            "optional": {
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 1024, "step": 1}),
            },
        }

    # 拿整个列表一次处理，避免 ComfyUI 逐张拆开串行执行
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "CausalTile"

    def run(self, llm, image, system_prompt, user_prompt, max_side, timeout=120, max_workers=4):
        llm = first_or_single(llm) or {}
        base_url = (llm.get("base_url") or "").strip()
        api_key = (llm.get("api_key") or "").strip()
        model = (llm.get("model") or "").strip()
        extra_headers = llm.get("headers") or {}

        if not base_url:
            raise RuntimeError("base_url 是空的，在 LLM Model Loader 里填一个 OpenAI 兼容的接口地址。")
        if not api_key:
            raise RuntimeError(
                "拿不到 api_key：可以直接填在节点里，也可以写 env:环境变量名，或设置 CAUSALTILE_API_KEY。"
            )

        if not model:
            models = fetch_models(base_url, api_key, extra_headers=extra_headers)
            if not models:
                raise RuntimeError("没有指定模型，而且接口没有返回任何可用模型。")
            model = models[0]
            print(f"[CausalTile] 未指定模型，自动使用：{model}")

        images = flatten_images(image)
        if not images:
            raise RuntimeError("LLM Image To Prompt：没有收到任何图片。")

        system_text = (first_or_single(system_prompt, "") or "").strip()
        user_text = (first_or_single(user_prompt, "") or "").strip() or DEFAULT_USER_PROMPT
        max_side = int(first_or_single(max_side, 1024) or 0)
        timeout = int(first_or_single(timeout, 120) or 120)
        workers = max(1, min(int(first_or_single(max_workers, 4) or 1), len(images)))

        def one(index):
            messages = []
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(images[index], max_side)}},
                    ],
                }
            )
            try:
                text = chat_completion(
                    base_url, api_key, model, messages,
                    timeout=timeout, extra_headers=extra_headers,
                )
            except Exception as exc:
                raise RuntimeError(f"第 {index + 1}/{len(images)} 张图反推失败：{exc}") from exc
            return strip_code_fence(text)

        started = time.time()
        if workers == 1 or len(images) == 1:
            prompts = [one(index) for index in range(len(images))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                prompts = list(pool.map(one, range(len(images))))

        print(
            f"[CausalTile] LLM 反推完成：{len(images)} 张图，并发 {workers}，"
            f"单请求超时 {timeout}s，耗时 {time.time() - started:.1f}s"
        )
        return (prompts,)
