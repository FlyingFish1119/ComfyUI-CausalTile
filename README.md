# ComfyUI-CausalTile

ComfyUI 的分块重绘节点。将大图切分为网格后，按从左到右、从上到下的顺序逐块采样重绘——每一块都会把左侧和上侧**刚刚重绘完成**的区域带入采样视野作为上下文参考，全程在 Latent 空间内裁剪、采样与混合，最后由下游节点统一解码。

![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-1f425f?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 效果预览

![CausalTile 成品示例](workflow/CausalTile_preview.png)

![CausalTile 成品示例](workflow/CausalTile_preview_realistic.png)


> 上图由 CausalTile 生成，PNG 文件内嵌了完整工作流（也可直接使用仓库中的 `workflow/CausalTile.json`），拖进 ComfyUI 即可加载。此处为缩略图（原图大小分别为 7680 * 4320 和 5760 * 4320）。

## 解决的问题与工作机制

常规的切块重绘流程在处理大图时存在三个结构性缺陷：

1. **分块之间彼此孤立**：每块独立裁剪、独立采样，相邻块之间没有上下文约束，重绘幅度稍大时交界处的色彩、光影和线条走向就容易脱节。
2. **始终以旧底图为参照**：后续分块裁剪时读取的仍是未经重绘的初始底图，无法感知前序分块已经生成的新内容。
3. **四向扩充视野引入未处理区域**：若将裁剪框向四周同步外扩，右侧和下侧尚未重绘的底图也会被带入采样视野，既占用采样面积，又会把待重绘区域的旧质感带进当前块。

**CausalTile 采用按行接力 + 单向扩充视野的方式处理分块：**

```text
┌─────────────────────────────────────────┐
│  view_box：送入采样器的实际视野           │
│  （左、上为已重绘完成的区域，权重为 0）   │
│                                         │
│            ┌──────┬─────────────────────┤
│            │ 羽化 │                     │
├────────────┼──────┼─────────────────────┤
│            │ 过渡 │                     │
│            │ 0→1  │      tile_box       │
│            │      │    （基础网格块）     │
│            │      │                     │
│            │      │   paint_box         │
│            │      │  （实际重绘区域）     │
└────────────┴──────┴─────────────────────┘
```

- **按行串行接力**：严格按从左到右、从上到下（行优先）的顺序逐块执行。前一块采样完成后立即按权重混合回主 Latent 画布，下一块直接从更新后的画布上裁剪视野，确保左、上两侧参考的永远是最新重绘结果。
- **单向几何规划（只向左、上扩展）**：
  每一块包含三个右下角对齐的矩形区域：
  - `tile_box`：画布按 `rows × columns` 均分出的基础网格块。
  - `paint_box`：在 `tile_box` 基础上向左、向上延伸重叠区（由 `overlap_ratio` 换算），构成实际参与重绘的区域。首行不向上延伸，首列不向左延伸，右、下边界不扩展。
  - `view_box`：在 `paint_box` 基础上继续向左、向上按 `view_factor` 倍数拉伸出的采样视野（超出画布左、上边界时截断到边界 `0`）。首行不向上拉伸，首列不向左拉伸。右侧与下侧直接对齐当前块边界，不纳入未重绘区域。
- **单向羽化遮罩**：
  在 `view_box` 内分别构建水平与垂直方向的一维线性渐变并相乘得到二维权重图：`paint_box` 起点之前的纯参考区域权重为 `0`；从 `paint_box` 起点开始，在设定的羽化宽度内由 `0` 线性递增至 `1`，其余区域保持为 `1`（首列水平方向全为 `1`，首行垂直方向全为 `1`）。右侧与下侧边缘不做羽化，交由后续相邻块覆盖，每条接缝只融合一次。
- **纯 Latent 空间合成与尺寸对齐**：
  - 拆图阶段会向接入的 `VAE` 查询空间压缩倍数 `align`（优先调用 `vae.spacial_compression_decode()`，其次读取 `vae.downscale_ratio`，默认回退为 `8`）。
  - 若原图宽高不能被 `columns × align` 或 `rows × align` 整除，规划时会自动向上取整并在右、下边缘复制填充（`replicate`），保证每块基础宽高及重叠区像素均为 `align` 的整数倍。
  - 采样与合成全程在 Latent 张量（兼容 4D `[B, C, H, W]` 与 5D `[B, C, T, H, W]`）上进行，全部块处理完毕后自动裁去右、下填充边角，交由外部节点一次性解码，避免分块独立解码造成的色差。

## 安装

进入 ComfyUI 的 `custom_nodes` 目录克隆本仓库，重启 ComfyUI：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FlyingFish1119/ComfyUI-CausalTile.git
```

节点位于右键菜单的 `CausalTile` 分类下，仅依赖 ComfyUI 环境自带的 `torch`、`Pillow`、`requests`。

## 基础连线

```text
  [底图]
    ├──> CausalTile 拆图 ──> tile_plan ──> CausalTile 设置提示词 ──> plan ─┐
    │          ↑ vae（读取压缩倍数）                                        │
    │                                                                      │
    └──> VAE Encode (Tiled) ──> latent ─────────────────────────────┐      │
                                                                    ↓      ↓
  [model] ──> Differential Diffusion ───────────────────> CausalTile 分块重绘
                                                                    │
                                                                    ↓ latent
                                                           VAE Decode (Tiled) ──> 最终图像
```

1. **缩放节点置于拆图之前**：CausalTile 不在分块内部做图像缩放，所有分块直接按输入底图的像素密度裁剪与采样。如需放大分辨率，请先通过放大节点（如 `Upscale Image (using Model)`、`ImageScale`）将底图缩放至目标尺寸，再同时输入给 `CausalTile 拆图` 和 `VAE Encode (Tiled)`。
2. **使用 Tiled VAE 编解码**：大图编码与解码建议使用 `VAE Encode (Tiled)` 和 `VAE Decode (Tiled)` 以控制显存峰值。接入 `CausalTile 拆图` 的 `vae` 仅用于探测压缩倍数，不执行编解码。
3. **配合 `Differential Diffusion` 使用**：在 `model` 进入 `CausalTile 分块重绘` 前串联 ComfyUI 内置的 `Differential Diffusion` 节点，可使单向羽化的 `noise_mask` 按连续权重控制重绘程度。

## 节点与参数说明

### 1. `CausalTile 拆图` (`CausalTileSplit`)

探测 VAE 压缩倍数、生成分块几何规划，并按行优先顺序切出各块的视野图像。

- **`image` (`IMAGE`)**：待重绘的底图。
- **`vae` (`VAE`)**：用于读取像素到 Latent 的空间压缩倍数。
- **`rows` / `columns` (`INT`，默认 `4`，范围 `1 ~ 16`)**：网格切分的行数与列数。
- **`overlap_ratio` (`FLOAT`，默认 `0.1`，范围 `0.0 ~ 1.0`)**：`paint_box` 相对基础块宽/高向左、向上延伸的重叠比例。内部按 `round(块尺寸 × overlap_ratio)` 换算为像素，并向上取整到 VAE 压缩倍数的整数倍。
- **`view_factor` (`FLOAT`，默认 `2.0`，范围 `1.0 ~ 4.0`)**：`view_box` 相对 `paint_box` 宽/高向左、向上拉伸的视野倍数，左上起点向下对齐到 VAE 压缩倍数的整数倍（触及画布左、上边界时截断为 `0`）。

**输出**：
- **`tile_plan` (`CAUSAL_TILE_PLAN`)**：分块几何规划对象，包含画布尺寸、对齐参数及所有分块的 `tile_box`、`paint_box`、`view_box` 坐标。
- **`tile_images` (`IMAGE` 列表)**：按行优先排列的各块图像列表。切出的是包含左、上邻居画面的 `view_box` 而非孤立的 `tile_box`，便于下游接入反推节点时获取局部画面在整图中的上下文。

### 2. `CausalTile 设置提示词` (`CausalTilePrompt`)

将提示词写入分块规划并返回新的 `plan` 对象（不原地修改原对象）。

- **`plan` (`CAUSAL_TILE_PLAN`)**：来自 `CausalTile 拆图` 的规划对象。
- **`prompts` (`STRING`)**：支持单条字符串或字符串列表。传入单条字符串时，所有分块复用该提示词；传入列表时，列表长度必须等于 `rows × columns`，按行优先顺序与各分块一一对应，数量不匹配时会抛出 `ValueError`。

### 3. `CausalTile 分块重绘` (`CausalTileDetailer`)

按 `tile_plan` 顺序逐块裁剪 Latent、构建遮罩、采样并加权贴回画布。

- **`latent` (`LATENT`)**：外部编码好的底图 Latent。若尺寸与规划画布不一致，节点会自动在右、下边缘补齐或裁切，并在全部块完成后裁回原图对应的 Latent 尺寸。
- **`tile_plan` (`CAUSAL_TILE_PLAN`)**：来自 `CausalTile 设置提示词` 的规划对象。
- **`model` / `clip`**：采样模型与文本编码器。单次运行内相同提示词的 CLIP 编码结果会自动缓存复用，不会重复编码。
- **`negative_prompt` / `seed` / `steps` / `cfg` / `sampler_name` / `scheduler`**：常规采样参数，其中第 `k` 块（从 `0` 开始计数）实际使用的随机种子为 `seed + k`。
- **`denoise` (`FLOAT`，默认 `0.35`，范围 `0.0 ~ 1.0`)**：内部块（`row > 0` 且 `col > 0`，即同时具备左、上两侧已重绘参考的块）的重绘幅度。
- **`first_denoise` (`FLOAT`，默认 `-1.0`，范围 `-1.0 ~ 1.0`)**：首块 `(0, 0)` 的重绘幅度。首块没有任何已重绘邻居作为参考；设为负数（如默认的 `-1.0`）时直接跟随 `denoise`。
- **`edge_denoise` (`FLOAT`，默认 `-1.0`，范围 `-1.0 ~ 1.0`)**：首行块（`row == 0, col > 0`，仅左侧有参考）与首列块（`row > 0, col == 0`，仅上侧有参考）的重绘幅度；设为负数时直接跟随 `denoise`。
- **`feather` (`INT`，默认 `16`，范围 `0 ~ 256`，步长 `4`)**：采样完成后贴回主画布时，左、上交界处的线性混合过渡宽度（单位为图像像素）。
- **`noise_mask_feather` (`INT`，默认 `16`，范围 `0 ~ 256`，步长 `4`)**：送入采样器的 `denoise_mask` 在左、上交界处的线性过渡宽度（单位为图像像素）。
  > `feather` 与 `noise_mask_feather` 在内部均会除以 VAE 压缩倍数转换为 Latent 像素，并与 `min(overlap_x, overlap_y) // compression` 取较小值（即实际生效的羽化宽度不会超过重叠区宽度，避免侵入基础块内部留下未重绘条带）。若需增大羽化过渡范围，需同步调大 `CausalTile 拆图` 中的 `overlap_ratio`。

### 附：LLM 局部提示词反推节点

仓库内另包含两个辅助节点 `LLM Model Loader` 与 `LLM Image To Prompt`（见 `workflow/CausalTile.json`）。可接入 OpenAI 兼容格式的多模态接口，将 `CausalTile 拆图` 输出的 `tile_images` 并发发送给模型反推各块的局部提示词列表，再传入 `CausalTile 设置提示词`。不使用分块独立提示词时无需连接。

## License

[MIT](LICENSE)
