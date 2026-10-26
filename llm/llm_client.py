"""CausalTile LLM 基础设施：密钥解析、请求头解析、模型发现、图片编码、聊天请求。"""

import base64
import io
import json
import os

import requests
import torch
from PIL import Image

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_models_cache.json")

# 关闭思考的字段，三级降级依次尝试：
#   1. 智谱 GLM 的写法 thinking.type=disabled
#   2. Qwen 系的写法 enable_thinking=false
#   3. 什么都不带，退回纯标准请求
# 服务端严格校验、拒绝未知参数时返回 400，chat_completion 自动降一级重发。
NO_THINKING_VARIANTS = (
    {"thinking": {"type": "disabled"}},
    {"enable_thinking": False},
    {},
)

TIMEOUT = 120


def resolve_api_key(raw):
    """支持 env:变量名 读环境变量；否则当作字面 key。"""
    raw = (raw or "").strip()
    if raw.startswith("env:"):
        return os.environ.get(raw[4:].strip(), "").strip()
    if raw:
        return raw
    return ""


def parse_headers(text):
    """把「一行一个 key: value」解析成 dict；空行和 # 开头的注释行跳过。"""
    headers = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, _, value = line.partition(":")
        name = name.strip()
        if name:
            headers[name] = value.strip()
    return headers


def _base(base_url):
    return (base_url or "").strip().rstrip("/")


def _headers(api_key, extra_headers=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def fetch_models(base_url, api_key, timeout=30, extra_headers=None):
    """GET {base_url}/models，返回模型 id 列表。"""
    resp = requests.get(
        f"{_base(base_url)}/models", headers=_headers(api_key, extra_headers), timeout=timeout
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"获取模型列表失败（HTTP {resp.status_code}）：{resp.text[:300]}")
    data = resp.json()
    items = data.get("data") if isinstance(data, dict) else None
    if items is None and isinstance(data, dict):
        items = data.get("models")
    if not isinstance(items, list):
        return []
    ids = []
    for item in items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("model")
        else:
            mid = item
        if mid:
            ids.append(str(mid))
    return sorted(set(ids))


def load_cached_models():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        models = data.get("models") if isinstance(data, dict) else None
        return [str(m) for m in models] if isinstance(models, list) else []
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"[CausalTile] 读取模型缓存失败：{exc}")
        return []


def save_cached_models(models):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as fp:
            json.dump({"models": list(models)}, fp, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[CausalTile] 写入模型缓存失败：{exc}")


def image_to_data_url(image, max_side):
    """把 IMAGE（[H, W, C] 的 0~1 float tensor）缩到最大边长内，编码成 data URL。"""
    pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    picture = Image.fromarray(pixels).convert("RGB")
    if max_side and max(picture.size) > max_side:
        picture.thumbnail((int(max_side), int(max_side)), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def chat_completion(base_url, api_key, model, messages, timeout=TIMEOUT, extra_headers=None):
    """POST {base_url}/chat/completions，返回首条回复文本。关思考字段按三级降级重试。"""
    url = f"{_base(base_url)}/chat/completions"
    headers = _headers(api_key, extra_headers)

    for extra in NO_THINKING_VARIANTS:
        payload = {"model": model, "messages": messages}
        payload.update(extra)
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code not in (400, 422):
            break

    if resp.status_code >= 400:
        raise RuntimeError(f"LLM 请求失败（HTTP {resp.status_code}）：{resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"LLM 返回结构异常：{json.dumps(data, ensure_ascii=False)[:500]}")


def strip_code_fence(text):
    """去掉模型习惯性加上的 ``` 代码围栏。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text[3:]
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()
