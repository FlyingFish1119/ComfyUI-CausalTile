"""分块几何规划、单向羽化遮罩与重绘采样。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers

try:
    import latent_preview
except ImportError:
    latent_preview = None

ALIGN = 8
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def align_up(value, align: int = ALIGN) -> int:
    v = int(value)
    return ((v + align - 1) // align) * align


def align_down(value, align: int = ALIGN) -> int:
    return (int(value) // align) * align


def first_or_single(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def pad_canvas_size(width: int, height: int, columns: int, rows: int,
                    align: int = ALIGN) -> Tuple[int, int]:
    # 保证切分后每块基础宽高都是 align 的倍数，避免 VAE 编解码时发生尺寸偏移。
    # align 就是 VAE 的压缩倍数，不是固定 8。
    return (
        align_up(width, columns * align),
        align_up(height, rows * align),
    )


def pad_image(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    _, h, w, _ = image.shape
    if w == width and h == height:
        return image
    bchw = image.movedim(-1, 1)
    bchw = F.pad(bchw, (0, width - w, 0, height - h), mode="replicate")
    return bchw.movedim(1, -1)



@dataclass
class CausalTile:
    index: int
    row: int
    col: int
    tile_box: Tuple[int, int, int, int]    # 基础网格 (x1, y1, x2, y2)
    paint_box: Tuple[int, int, int, int]   # 重绘区，向左上扩 paint_overlap
    view_box: Tuple[int, int, int, int]    # 采样视野区，向左上按 view_factor 拉伸
    prompt: str = ""


@dataclass
class CausalTilePlan:
    width: int
    height: int
    pad_width: int
    pad_height: int
    rows: int
    columns: int
    overlap_x: int
    overlap_y: int
    view_factor: float
    align: int
    tiles: List[CausalTile]


def ratio_to_overlap(size: int, ratio: float, align: int = ALIGN) -> int:
    """块宽/高 × 比例 → 像素，向上取整到 align 的倍数。"""
    return align_up(max(0, int(round(size * ratio))), align)


def plan_tiles(width: int,
               height: int,
               rows: int,
               columns: int,
               overlap_ratio: float,
               view_factor: float,
               align: int = ALIGN) -> CausalTilePlan:
    pad_width, pad_height = pad_canvas_size(width, height, columns, rows, align)
    bw = pad_width // columns
    bh = pad_height // rows

    # 比例换算成像素，向上取整并按 align 对齐：小比例也不会被截成 0 像素，边界又能对上 VAE 的压缩倍数
    overlap_x = ratio_to_overlap(bw, overlap_ratio, align)
    overlap_y = ratio_to_overlap(bh, overlap_ratio, align)

    tiles: List[CausalTile] = []
    for r in range(rows):
        for c in range(columns):
            bx1, by1 = c * bw, r * bh
            bx2, by2 = bx1 + bw, by1 + bh

            # 左、上有已重绘邻居时才向外延伸 overlap
            gx1 = bx1 if c == 0 else max(0, bx1 - overlap_x)
            gy1 = by1 if r == 0 else max(0, by1 - overlap_y)
            gx2, gy2 = bx2, by2

            gw, gh = gx2 - gx1, gy2 - gy1

            # 视野框只向左上扩上下文，右下与基础块对齐
            if c == 0:
                yx1 = gx1
            else:
                target_w = int(round(gw * view_factor))
                yx1 = align_down(max(0, gx2 - target_w), align)
            if r == 0:
                yy1 = gy1
            else:
                target_h = int(round(gh * view_factor))
                yy1 = align_down(max(0, gy2 - target_h), align)
            yx2, yy2 = gx2, gy2

            tiles.append(CausalTile(
                index=len(tiles),
                row=r,
                col=c,
                tile_box=(bx1, by1, bx2, by2),
                paint_box=(gx1, gy1, gx2, gy2),
                view_box=(yx1, yy1, yx2, yy2),
            ))

    return CausalTilePlan(
        width=width,
        height=height,
        pad_width=pad_width,
        pad_height=pad_height,
        rows=rows,
        columns=columns,
        overlap_x=overlap_x,
        overlap_y=overlap_y,
        view_factor=view_factor,
        align=align,
        tiles=tiles,
    )


def apply_prompts(plan: CausalTilePlan, prompts) -> CausalTilePlan:
    """将提示词写入分块（单条则全图复用，列表则按行优先一一对应）。"""
    total = len(plan.tiles)
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [str(p) for p in prompts]

    if len(prompts) == 1:
        prompts = prompts * total
    elif len(prompts) != total:
        raise ValueError(
            f"CausalTilePrompt: 提示词数量 {len(prompts)} 与分块数 {total} 不匹配，"
            f"请让列表长度等于分块数，或只传一条全局提示词"
        )

    new_tiles = [
        CausalTile(
            index=tile.index,
            row=tile.row,
            col=tile.col,
            tile_box=tile.tile_box,
            paint_box=tile.paint_box,
            view_box=tile.view_box,
            prompt=prompt,
        )
        for tile, prompt in zip(plan.tiles, prompts)
    ]

    return CausalTilePlan(
        width=plan.width,
        height=plan.height,
        pad_width=plan.pad_width,
        pad_height=plan.pad_height,
        rows=plan.rows,
        columns=plan.columns,
        overlap_x=plan.overlap_x,
        overlap_y=plan.overlap_y,
        view_factor=plan.view_factor,
        align=plan.align,
        tiles=new_tiles,
    )


def _ramp_1d(total: int, start: int, feather: int) -> torch.Tensor:
    """[0, start) 为 0，[start, start+feather) 线性过渡到 1，其余为 1。"""
    t = torch.zeros(int(total), dtype=torch.float32)
    if start <= 0:
        t.fill_(1.0)
        return t

    end = min(int(start) + int(feather), int(total))
    if end > start:
        t[start:end] = torch.linspace(0.0, 1.0, end - start + 2)[1:-1]
    if end < total:
        t[end:] = 1.0
    return t


def build_mask(width: int,
               height: int,
               start_x: int,
               start_y: int,
               feather_x: int,
               feather_y: int) -> torch.Tensor:
    """生成单向羽化遮罩 [1, height, width]，start_x/start_y 为重绘区起点。"""
    rx = _ramp_1d(width, start_x, feather_x)
    ry = _ramp_1d(height, start_y, feather_y)
    mask = ry[:, None] * rx[None, :]
    return mask.unsqueeze(0)


def vae_compression(vae) -> int:
    """VAE 的像素→latent 压缩倍数，优先问 VAE 本身，问不到按 8 算。"""
    getter = getattr(vae, "spacial_compression_decode", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    ratio = getattr(vae, "downscale_ratio", 0)
    return int(ratio) if ratio else ALIGN


def encode_text(clip, text: str):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text or ""))


def _mem(tag: str) -> None:
    """临时诊断：打印当前显存占用、峰值和缓存池大小。"""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[CausalTile][mem] {tag}: alloc {alloc:.2f}G / peak {peak:.2f}G / reserved {reserved:.2f}G")


def sample_latent(model,
                  positive,
                  negative,
                  latent: dict,
                  seed: int,
                  steps: int,
                  cfg: float,
                  sampler_name: str,
                  scheduler: str,
                  denoise: float) -> torch.Tensor:
    samples = latent["samples"]
    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        noise_mask = noise_mask.to(device=samples.device, dtype=samples.dtype)

    noise = comfy.sample.prepare_noise(samples, int(seed) & MAX_SEED, None)
    sampler = comfy.samplers.KSampler(
        model,
        steps=int(steps),
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=float(denoise),
        model_options=model.model_options,
    )
    callback = None
    if latent_preview is not None:
        callback = latent_preview.prepare_callback(model, int(steps))
    return sampler.sample(
        noise,
        positive,
        negative,
        cfg=float(cfg),
        latent_image=samples,
        denoise_mask=noise_mask,
        callback=callback,
        seed=int(seed) & MAX_SEED,
    )


def fit_latent_to_plan(latent: torch.Tensor, expect_h: int, expect_w: int) -> torch.Tensor:
    """把外部编码好的 latent 对齐到画布需要的尺寸：不够在右、下补边，多了裁掉。"""
    h, w = int(latent.shape[-2]), int(latent.shape[-1])
    if (h, w) == (expect_h, expect_w):
        return latent
    pad_right = max(0, expect_w - w)
    pad_bottom = max(0, expect_h - h)
    if pad_right or pad_bottom:
        print(
            f"[CausalTile] latent {w}x{h} 比画布 {expect_w}x{expect_h} 小，"
            f"在右、下补 {pad_right}x{pad_bottom} 个 latent 像素",
            flush=True,
        )
        latent = F.pad(latent, (0, pad_right, 0, pad_bottom), mode="replicate")
    elif w > expect_w or h > expect_h:
        print(
            f"[CausalTile] latent {w}x{h} 比画布 {expect_w}x{expect_h} 大，裁掉多余部分",
            flush=True,
        )
    return latent[..., :expect_h, :expect_w].contiguous()


def run_causal_detail(latent_canvas: torch.Tensor,
                      plan: CausalTilePlan,
                      model,
                      clip,
                      negative_prompt,
                      seed: int,
                      steps: int,
                      cfg: float,
                      sampler_name: str,
                      scheduler: str,
                      denoise: float,
                      first_denoise: float,
                      edge_denoise: float,
                      feather: int,
                      noise_mask_feather: int) -> torch.Tensor:
    """按行优先顺序逐块采样，全程在传入的 latent 画布上裁剪与贴回。

    编码这一步交给外部节点，压缩倍数由 plan 带进来（来自拆图时探测的 vae），这里只照着它切坐标。
    """
    compression = int(getattr(plan, "align", ALIGN))

    expect_h = plan.pad_height // compression
    expect_w = plan.pad_width // compression
    latent_canvas = fit_latent_to_plan(latent_canvas.clone(), expect_h, expect_w)
    latent_rank = latent_canvas.ndim
    total_tiles = len(plan.tiles)

    # 羽化宽度不能超过 overlap，否则会侵入基础块内部留下未重绘条带。
    # latent 里一个像素就是 compression 个像素，所以羽化颗粒度也按它取整。
    max_feather = min(plan.overlap_x, plan.overlap_y) // compression
    feather = max(0, min(int(feather) // compression, max_feather))
    noise_mask_feather = max(0, min(int(noise_mask_feather) // compression, max_feather))

    started = time.perf_counter()
    print(
        f"[CausalTile] 画布 latent {tuple(latent_canvas.shape)}，共 {total_tiles} 块待重绘",
        flush=True,
    )
    _mem("进入 detailer")
    torch.cuda.reset_peak_memory_stats()

    def crop(tensor, lx1, ly1, lx2, ly2):
        # latent 可能是 4 维 [B, C, H, W]，也可能是视频式的 5 维 [B, C, T, H, W]，都按末尾两维切
        return tensor[..., ly1:ly2, lx1:lx2]

    def broadcast(mask):
        # [1, h, w] 补到能和任意维度的 latent 广播
        return mask.reshape((1,) * (latent_rank - 2) + tuple(mask.shape[-2:]))

    cond_cache = {}

    def _positive(text):
        key = text or ""
        if key not in cond_cache:
            cond_cache[key] = encode_text(clip, key)
        return cond_cache[key]

    negative = encode_text(clip, negative_prompt or "")

    for tile in plan.tiles:
        tile_started = time.perf_counter()
        # 首块无上下文、首行首列只有半边上下文，各自可用独立的 denoise
        if tile.index == 0:
            tile_denoise = first_denoise if first_denoise >= 0 else denoise
        elif tile.row == 0 or tile.col == 0:
            tile_denoise = edge_denoise if edge_denoise >= 0 else denoise
        else:
            tile_denoise = denoise

        yx1, yy1, yx2, yy2 = tile.view_box
        gx1, gy1, _, _ = tile.paint_box

        lx1, ly1 = yx1 // compression, yy1 // compression
        lx2, ly2 = yx2 // compression, yy2 // compression
        lw, lh = lx2 - lx1, ly2 - ly1
        rx1, ry1 = gx1 // compression - lx1, gy1 // compression - ly1

        latent = {
            "samples": crop(latent_canvas, lx1, ly1, lx2, ly2).contiguous(),
            "noise_mask": broadcast(
                build_mask(lw, lh, rx1, ry1, noise_mask_feather, noise_mask_feather)
            ),
        }

        pos = _positive(tile.prompt)
        sampled = sample_latent(
            model, pos, negative, latent,
            int(seed) + tile.index,
            steps, cfg, sampler_name, scheduler, tile_denoise,
        )

        base = crop(latent_canvas, lx1, ly1, lx2, ly2)
        sampled = sampled.to(device=base.device, dtype=base.dtype)
        weight = broadcast(build_mask(lw, lh, rx1, ry1, feather, feather)).to(
            device=base.device, dtype=base.dtype
        )
        latent_canvas[..., ly1:ly2, lx1:lx2] = base * (1.0 - weight) + sampled * weight

        done = tile.index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (total_tiles - done)
        print(
            f"[CausalTile] 块 {done}/{total_tiles} ({tile.row},{tile.col}) 完成，"
            f"view {lw}x{lh} latent，denoise {tile_denoise:.2f}，"
            f"本块 {time.perf_counter() - tile_started:.1f}s，"
            f"累计 {elapsed:.0f}s，预计剩余 {eta:.0f}s",
            flush=True,
        )
        _mem(f"块 {tile.index} 显存")

    # 把 padding 出来的边角裁掉。latent 粒度只能向上取整，保证覆盖原始像素范围
    keep_w = min(int(latent_canvas.shape[-1]), (int(plan.width) + compression - 1) // compression)
    keep_h = min(int(latent_canvas.shape[-2]), (int(plan.height) + compression - 1) // compression)
    latent_canvas = latent_canvas[..., :keep_h, :keep_w].contiguous()

    print(f"[CausalTile] latent {tuple(latent_canvas.shape)} 合成完成，交由外部节点解码")
    _mem("全部块结束")
    return latent_canvas
