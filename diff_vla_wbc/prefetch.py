"""CPU 侧预取 / 步骤内缓存，给 GR00T VLA 的 ``processed_batch`` 用。

背景（2026-10-25 卡 6 实测）：联合微调每步 GPU 只忙 ~0.6 s，但整步要 ~2.0 s —— 瓶颈在
CPU 的 ``TaskObservationBuilder.build``（mp4 随机访问 ~40 ms/帧）+ ``Gr00tN1d7Processor``
（图像 transform / tokenize），而 GPU 只能在旁边等。更糟的是**同一步里 ``processed_batch``
被算了两遍**：``provider.sample()`` 里的 token 采样一次、``train_step`` 里的手部采样再一次。

两个工具：

* :class:`StepMemo` —— 步骤内缓存：同一组 ``frames`` 只处理一次（一次 forward 里给 token 和
  手部两路复用），步结束时 ``clear()``。
* :class:`BatchPrefetcher` —— 跨步预取：主线程算上一步的 backward 时，后台线程把下一步的
  ``processed_batch`` 先做出来（只在 CPU 上做，**不碰 CUDA**，避免跨线程用 CUDA stream）。

两者都是纯 Python/线程，不改任何数值路径：命中缓存与现算的结果必须逐位相同（见
``scripts/check_prefetch_parity.py``）。
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Hashable


def frames_key(frames: Any) -> Hashable:
    """把 ``frames``（tensor / list / tuple）变成稳定的缓存键。"""
    try:
        if hasattr(frames, "detach"):
            frames = frames.detach().cpu().tolist()
        if isinstance(frames, (int,)):
            return (int(frames),)
        return tuple(int(f) for f in frames)
    except Exception:
        return ("raw", id(frames))


class StepMemo:
    """步骤内的一次性缓存（``frames`` -> 处理结果）。"""

    def __init__(self, fn: Callable[[Any], Any], max_entries: int = 8):
        self._fn = fn
        self._max = int(max_entries)
        self._items: dict[Hashable, Any] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.calls = 0

    def get(self, frames: Any) -> Any:
        key = frames_key(frames)
        with self._lock:
            if key in self._items:
                self.hits += 1
                return self._items[key]
        value = self._fn(frames)
        with self._lock:
            self.calls += 1
            if len(self._items) < self._max:
                self._items[key] = value
        return value

    def put(self, frames: Any, value: Any) -> None:
        key = frames_key(frames)
        with self._lock:
            if len(self._items) < self._max:
                self._items[key] = value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"calls": self.calls, "hits": self.hits, "entries": len(self._items)}


class BatchPrefetcher:
    """单 worker 线程预取：``submit(frames)`` 后，``get(frames)`` 取结果。

    * 只预取 CPU 侧处理（``processed_batch``），**CUDA 前向仍在主线程**；
    * ``pending`` 上限 2，避免把内存堆爆；
    * 任何异常都会在 ``get`` 时原样抛出（不会静默降级成错误结果）。
    """

    def __init__(self, fn: Callable[[Any], Any], pending: int = 2, enabled: bool = True):
        self._fn = fn
        self._pending_max = max(1, int(pending))
        self.enabled = bool(enabled)
        self._slots: dict[Hashable, Any] = {}
        self._order: list[Hashable] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[tuple[Hashable, Any]] = []
        self._errors: dict[Hashable, BaseException] = {}
        self.hits = 0
        self.computed = 0
        self._stop = False
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="vla-prefetch", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop and not self._queue:
                    return
                key, frames = self._queue.pop(0)
            try:
                value = self._fn(frames)
                err = None
            except BaseException as exc:  # noqa: BLE001 - 原样回抛给主线程
                value, err = None, exc
            with self._cv:
                if err is None:
                    self._slots[key] = value
                    self.computed += 1
                else:
                    self._errors[key] = err
                self._order.append(key)
                self._cv.notify_all()

    # ------------------------------------------------------------------ API
    def submit(self, frames: Any) -> None:
        if not self.enabled:
            return
        key = frames_key(frames)
        with self._cv:
            if key in self._slots or key in self._errors or any(k == key for k, _ in self._queue):
                return
            while len(self._order) + len(self._queue) >= self._pending_max and self._order:
                old = self._order.pop(0)
                self._slots.pop(old, None)
                self._errors.pop(old, None)
            self._queue.append((key, frames))
            self._cv.notify_all()

    def get(self, frames: Any, fallback: Callable[[], Any]) -> Any:
        """取预取结果；没有就现算（``fallback``）。"""
        if not self.enabled:
            return fallback()
        key = frames_key(frames)
        with self._cv:
            while True:
                if key in self._errors:
                    err = self._errors.pop(key)
                    raise err
                if key in self._slots:
                    self.hits += 1
                    value = self._slots.pop(key)
                    if key in self._order:
                        self._order.remove(key)
                    return value
                if any(k == key for k, _ in self._queue):
                    self._cv.wait(timeout=5.0)
                    continue
                break
        # 未预取到：现算并登记（避免重复算）
        value = fallback()
        with self._cv:
            self._slots[key] = value
            self._order.append(key)
            while len(self._order) > self._pending_max:
                old = self._order.pop(0)
                self._slots.pop(old, None)
        return value

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def stats(self) -> dict[str, int]:
        with self._cv:
            return {"prefetched": self.computed, "hits": self.hits, "pending": len(self._slots)}
