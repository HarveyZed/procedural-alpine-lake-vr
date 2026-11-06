# VR 全景移植手册（engine_vr / vr.py）

本仓库是 [procedural-alpine-lake](https://github.com/Leonxlnx/procedural-alpine-lake) 的衍生移植：把原 OpenGL 4.3 管线完整移植到 **Apple Silicon（MoltenVK → Metal）**，并把成片形态从平面电影改为 **360° VR 全景**（等距柱状投影静帧 / SBS 立体 / 48fps 全景视频）。原 `engine/`、`audio/`、`docs/` 保持只读；移植代码全部在 `engine_vr/` + `vr.py` + `package_vr.py`。

> **授权提示**：上游仓库未附 LICENSE（默认保留所有权利）。本移植以学习研究为目的开源于此；商用请先联系原作者 Leon Lin。

## 1. 环境（一次）

- Apple Silicon Mac，macOS ≥ 14；磁盘 ≥ 10 GB 工作区。
- `uv`（或 pip）建 venv：`uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python numpy scipy pillow vulkan ruff`
  - 沙箱环境注意 `UV_CACHE_DIR=$PWD/tools/.uvcache`。
- 工作区本地 Vulkan 工具链引导（glslc / spirv-val / MoltenVK，均不落系统）见 `port/PHASE_0.md`；产物在 `tools/vk/`。
- **每个新 shell 必须先** `source tools/vk/env.sh`（设置 VKSDK_LOCAL / DYLD_LIBRARY_PATH / VK_DRIVER_FILES）。

## 2. 烘焙世界（GPU，约几分钟；缓存约 4 GB，可删可重烘）

按 `port/RUN_FIRST.md` §1–§3 的阶梯执行：先 512² 冒烟烘（`/tmp/p5_mini`），再全量 `run_eroded_bake()`。要点：

```bash
source tools/vk/env.sh
export ALPINE_CACHE=$PWD/cache_final   # 目录需先 mkdir -p
export ALPINE_MAX_PROC_GB=64           # 8192x4096 渲染改 96
export ALPINE_MIN_COMMIT_GB=24 ALPINE_MAX_GPU=1
```

烘完跑视点自检：`.venv/bin/python engine_vr/check_viewpoints.py --cache "$PWD/cache_final" --nominal-ok`。

## 3. 渲染

| 目标 | 命令（前缀均为 `ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py`） | 实测耗时 |
|---|---|---|
| 单目全景静帧 | `--viewpoint 0 --res 8192x4096 --spp 4 --out pano.png` | ~220 s @1spp，RSS 19 GiB |
| SBS 立体 | 加 `--stereo sbs`（每眼宽=总宽/2） | ~200 s @1spp 4096/眼 |
| 镜头动画帧 | `--animate --shot skim --dur 6.5 --fps 48 --spp 12 --frames-dir DIR`（180° 快门内置，逐样本相机+uTime） | ~23–73 s/帧随镜头 |

- 9 个视点 = 原片 9 镜头：`forest reveal skim waterfall stream meadow flyover peaks finale`。
- 断点续渲：帧目录已有帧自动跳过；先 `package_vr.py --dry-run` 看打包与编码命令。
- 吞吐量参考：约 0.52 s/Mspp（M5 Ultra）。整片 60 s 9 镜头 @48fps/12spp ≈ 28 h。

## 4. 打包上传（B站/YouTube 全景）

```bash
# 1) 帧 -> mp4：用 vr.py 打印的 ENCODE 命令行原样执行（自带 equirect/stereo 元数据）
# 2) 补音轨或静音轨（视频流不动）：
ffmpeg -i in.mp4 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
  -c:v copy -c:a aac -b:a 320k -shortest -movflags +faststart tmp.mp4
# 3) 全景元数据注入（最后一步，它会重写 moov）：
PYTHONPATH=tools/spatial-media .venv/bin/python tools/spatial-media/run_py312.py -i tmp.mp4 out.mp4
# 4) 双验：ffprobe 双流 + 同上传入 out.mp4 应报 Spherical/equirectangular
```

- B站规格自检线：mp4 / hevc / yuv420p / 分辨率 ≤ 8192x4320 / 4K 码率 > 20 Mbps / ≤ 120 fps / **必须有音轨**。
- 立体样例用 YouTube SBS 规格（每眼正方形，top-bottom 备选）；B站只吃立体声，不支持空间音频。
- 声音：`audio/soundtrack.py --stems`（环境声 ~5 s）→ `audio/score_c.py`（配乐+总混 ~20 s）→ `out/audio/soundtrack.wav`（60.000 s / -14 LUFS，与镜头切点原生同步）。

## 5. 验证与文档地图

- 回归门：`port/gates/gate_p1..p5.py`（常态可跑，全绿=可发布）；`gate_p0/p6/p7/p8/p12` 是时点评审门（无 diff 时会按设计报错）。
- 移植全程记录：`port/RUNBOOK.md`（契约与决策 D1–D8）、`port/PHASE_0..12.md`（各阶段任务书）、`port/reports/`（施工与评审报告）、`port/RUN_FIRST.md`（首次运行阶梯与故障特征表——**跑挂先查它**）。
- 已知边界：8192 全景需要 P6 守卫（NaN 出逃/步数上限已内置，SPIR-V 与 film 路径字节一致）；FOA 空间音频未做；`vr.py` 尚有存量 lint 告警。

## 6. 交付物样例（本仓库不打包，产物目录 `engine_vr/out/`）

mono 8192x4096 PNG、SBS 8192x2048 PNG、转台 4096x2048 SBS mp4、60 s 正片（48fps 12spp 带声 AAC）与 B站注入版。
