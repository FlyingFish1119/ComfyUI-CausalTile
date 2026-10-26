import asyncio

from aiohttp import web

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from .llm_client import fetch_models, parse_headers, resolve_api_key, save_cached_models


def register_routes():
    """把模型发现接口挂到 ComfyUI 的 HTTP 服务上，供节点上的刷新按钮调用。"""
    if PromptServer is None:
        print("[CausalTile] 不在 ComfyUI 服务进程中，模型发现接口未注册。")
        return

    try:
        routes = PromptServer.instance.routes
    except Exception as exc:
        print(f"[CausalTile] 模型发现接口注册失败，刷新按钮将不可用：{exc}")
        return

    @routes.post("/causaltile/llm/models")
    async def list_models(request):
        body = await request.json()
        base_url = (body.get("base_url") or "").strip()
        api_key = resolve_api_key(body.get("api_key"))
        extra_headers = parse_headers(body.get("headers"))
        if not base_url:
            return web.json_response({"error": "base_url 不能为空"}, status=400)
        try:
            models = await asyncio.to_thread(fetch_models, base_url, api_key, 30, extra_headers)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        save_cached_models(models)
        return web.json_response({"models": models})
