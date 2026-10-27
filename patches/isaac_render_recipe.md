# patch: 本机 Isaac 渲染可用配方（3 个环境开关，缺一不可）

> 背景：本机（2×RTX 4090 / 驱动 570.124.06）CUDA 正常，但 Isaac 的渲染器一直报
> `Graphics plugins not available` / `No device could be created` / `GLXBadFBConfig`。
> 逐个排查后确认**不是硬件、不是驱动版本**，而是三个环境开关。2026-10-27 验证通过：
>
> ```
> [cam] app OK enable_cameras=True
> [cam] rgb=(1, 64, 64, 3) mean=10.25        ← 真实渲染出的帧
> ```

## 一、根因（三条，都是"缺一个就失败"）

| # | 现象 | 根因 | 开关 |
|---|---|---|---|
| 1 | 无 X：`vkCreateInstance` 失败、`No device could be created` | 这个容器的 **NVIDIA Vulkan ICD 需要 X 连接**才能初始化 | `Xvfb :99` + `DISPLAY=:99`（+ `XDG_RUNTIME_DIR`）|
| 2 | 有 X：`X Error: GLXBadFBConfig` 进程直接死 | Xvfb 的 GLX（llvmpipe）默认只报低版本 GL，GLFW 要 4.x fbconfig 拿不到 | `MESA_GL_VERSION_OVERRIDE=4.6`、`MESA_GLSL_VERSION_OVERRIDE=460` |
| 3 | 用 `vglrun`：`libEGL.so.1` 找不到 / `carb.launcher` 刷一屏 env | VirtualGL 的 `LD_PRELOAD` 注入污染了 Kit 的子进程 | **不要用 VirtualGL**；`unset LD_PRELOAD` |

额外确认：`CUDA_VISIBLE_DEVICES` 必须 **unset**（carb 不认，会判 `CUDA in bad state` 或找不到 GPU）。

## 二、可用配方（复制即用）

```bash
# 1) 虚拟显示（只需一次；常驻）
setsid nohup Xvfb :99 -screen 0 1920x1080x24 > /tmp/xvfb.log 2>&1 &
mkdir -p /tmp/xdg && chmod 700 /tmp/xdg

# 2) 环境
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
source scripts/isaac_env.sh
unset CUDA_VISIBLE_DEVICES
unset LD_PRELOAD                      # ← 有 virtualgl 装过时必须清
export DISPLAY=:99
export XDG_RUNTIME_DIR=/tmp/xdg
export TERM=xterm
export MESA_GL_VERSION_OVERRIDE=4.6
export MESA_GLSL_VERSION_OVERRIDE=460

# 3) 直接用 venv python（不要 isaaclab.sh：TERM=dumb 下 `tabs` 会失败）
$ISAAC_VENV/bin/python -u <你的脚本>.py --headless --enable_cameras
```

## 三、判定"渲染真的通了"的最小检查

```python
# 建一个 Camera（CameraCfg, data_types=["rgb"]），reset 后调用 camera.update(dt)
rgb = cam.data.output["rgb"]        # 期望 (1,64,64,3)，且 mean 明显 > 0
```

* `mean ≈ 0` ⇒ 只建了相机但没渲染（通常是开关 1 或 2 没给）；
* `GLXBadFBConfig` ⇒ 开关 2；
* `vkCreateInstance` / `No device could be created` ⇒ 开关 1。

## 四、已知的"假故障"（不要被它们误导）

| 日志 | 是否致命 |
|---|---|
| `[gpu.foundation.plugin] The chosen activeGpu index 0 is higher than the available GPUs` | ❌ 非致命（历史上闭环能跑完时也有）|
| `[omni.physx.foundation.plugin] Unable to get IGpuFoundation, GpuDevices or Graphics` | ❌ 非致命（物理走 CPU）|
| `[carb.launcher.plugin] failed to fork the child process ... errno 11` | ⚠️ 检查是否 `LD_PRELOAD` 被注入 |
| `[cam] rgb ... mean=10.25` 但 GPU 利用率低 | 正常（headless 小分辨率）|

## 五、性能参考（本机实测）

| 项 | 值 |
|---|---|
| Isaace 启动 | 62–106 s（一次性）|
| 物理吞吐（256 物体、无渲染）| **15,281 env-transitions/s** |
| 单环境 51–60 步/s | 带 render 的老 bench；纯物理更快 |
| 64×64 相机采样 | 毫秒级（不是瓶颈；VLA 推理才是）|
