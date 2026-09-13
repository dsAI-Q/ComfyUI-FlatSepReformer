# -*- coding: utf-8 -*-
"""
ComfyUI-FlatSepReformer
=======================

FLASepformer（FLA-SepReformer-B）双说话人语音分离模型的 ComfyUI 自定义节点。

模型: iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100
  - 任务: speech-separation（8 kHz 单声道双说话人语音分离）
  - 训练集: Libri2Mix-100
  - 模型文件需放置在 <ComfyUI>/models/FlatSepReformer/ 下
    （运行 python install.py 自动下载，或手动从 ModelScope 拷贝）

节点设计（v2 合并版）:
  * FlatSepReformerSeparate : 合并"模型加载 + 语音分离"为单节点，
    自动探测 <ComfyUI>/models/FlatSepReformer（基于插件位置的相对路径推导），
    无需手动指定模型路径、无需额外加载节点。
  * FlatSepReformerLoadAudio / FlatSepReformerSaveAudio : 音频加载/保存。

推理后端:
  * onnx       : ONNX Runtime（onnx_model.onnx），无需 modelscope，默认优先
  * modelscope : ModelScope pipeline（pytorch_model.pt），需 master 源码版
  * auto       : 优先 onnx，缺失时回退 modelscope

AUDIO 类型遵循社区通用约定:
  {"waveform": torch.Tensor [B, C, T], "sample_rate": int}
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

MODEL_ID = "iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
MODEL_DIR_NAME = "FlatSepReformer"
TARGET_SAMPLE_RATE = 8000
NUM_SPEAKERS = 2
MODEL_FILES = ["configuration.json", "pytorch_model.pt", "onnx_model.onnx"]
MODELSCOPE_BASE = f"https://modelscope.cn/models/{MODEL_ID}/resolve/master"

# 模块级模型缓存: {(model_dir, backend, device): model}
_MODEL_CACHE = {}


# --------------------------------------------------------------------------- #
# 路径探测（相对插件位置推导，不依赖工作目录）
# --------------------------------------------------------------------------- #
def _plugin_parents() -> tuple[str, str]:
    """返回 (custom_nodes 目录, ComfyUI 根目录)。"""
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)          # custom_nodes
    grandparent = os.path.dirname(parent)   # ComfyUI 根
    return parent, grandparent


def find_model_dir(explicit_dir: str = "") -> str:
    """定位模型目录。

    默认: <ComfyUI>/models/FlatSepReformer
    （插件位于 <ComfyUI>/custom_nodes/ComfyUI-FlatSepReformer 时，
     即相对插件目录 ../../models/FlatSepReformer）

    优先级:
      1. 环境变量 FLATSEPREFORMER_MODEL_DIR
      2. 相对插件位置推导 <ComfyUI>/models/FlatSepReformer
      3. 插件内 models/ 与工作目录 models/
    """
    if explicit_dir and os.path.isdir(explicit_dir):
        return os.path.abspath(explicit_dir)

    candidates: list[str] = []

    env_dir = os.environ.get("FLATSEPREFORMER_MODEL_DIR", "")
    if env_dir:
        candidates.append(env_dir)

    parent, grandparent = _plugin_parents()
    if os.path.basename(parent) == "custom_nodes":
        candidates.append(os.path.join(grandparent, "models", MODEL_DIR_NAME))
    candidates.append(os.path.join(parent, "models", MODEL_DIR_NAME))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", MODEL_DIR_NAME))
    candidates.append(os.path.join(os.getcwd(), "models", MODEL_DIR_NAME))

    for c in candidates:
        c = os.path.abspath(c)
        if not os.path.isdir(c):
            continue
        files = os.listdir(c)
        if any(f in files for f in ("configuration.json", "pytorch_model.pt", "onnx_model.onnx")):
            return c
    return ""


def _default_output_dir() -> str:
    parent, grandparent = _plugin_parents()
    root = grandparent if os.path.basename(parent) == "custom_nodes" else os.getcwd()
    out = os.path.join(root, "output")
    os.makedirs(out, exist_ok=True)
    return out


def _default_model_dir() -> str:
    """模型默认目标文件夹: <ComfyUI>/models/FlatSepReformer（相对插件路径推导）。"""
    parent, grandparent = _plugin_parents()
    if os.path.basename(parent) == "custom_nodes":
        return os.path.join(grandparent, "models", MODEL_DIR_NAME)
    return os.path.join(os.getcwd(), "models", MODEL_DIR_NAME)


def _auto_download_model(model_dir: str) -> None:
    """运行时自动下载模型（HTTP 直连 ModelScope，无需安装 modelscope 包）。

    下载 <ComfyUI>/models/FlatSepReformer/ 下的三个模型文件
    （configuration.json / pytorch_model.pt / onnx_model.onnx）。
    """
    if os.path.isfile(os.path.join(model_dir, "configuration.json")):
        return
    os.makedirs(model_dir, exist_ok=True)
    import urllib.request
    for fname in MODEL_FILES:
        dest = os.path.join(model_dir, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            continue
        print(f"[FlatSepReformer] 正在自动下载模型文件: {fname} ...")
        try:
            urllib.request.urlretrieve(f"{MODELSCOPE_BASE}/{fname}", dest)
        except Exception as e:
            raise RuntimeError(
                f"模型自动下载失败（{fname}）: {e}\n"
                f"请手动下载模型后放入目标文件夹: {os.path.abspath(model_dir)}\n"
                "下载方式: 夸克网盘 https://pan.quark.cn/s/33060e1ee34c ；"
                "魔塔 https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
            ) from e
    print(f"[FlatSepReformer] 模型下载完成: {os.path.abspath(model_dir)}")


# --------------------------------------------------------------------------- #
# 音频工具
# --------------------------------------------------------------------------- #
def _to_mono_float32(audio: dict) -> tuple[np.ndarray, int]:
    """从 AUDIO dict 提取 [T] float32 mono 波形和采样率。"""
    if torch is None or audio is None:
        raise RuntimeError("torch 不可用，无法处理 AUDIO 数据（请确认在 ComfyUI 中运行）")
    waveform = audio.get("waveform")
    sr = int(audio.get("sample_rate", TARGET_SAMPLE_RATE))
    if waveform is None:
        raise ValueError("AUDIO 缺少 waveform 字段")
    w = waveform.detach().cpu().numpy()          # [B, C, T]
    if w.ndim == 1:
        w = w[None, None, :]
    elif w.ndim == 2:
        w = w[None, :, :]
    w = w[0, 0]                                   # 取第一个样本、第一个声道
    return w.astype(np.float32), sr


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """重采样到目标采样率。优先 scipy，回退 numpy 线性插值。"""
    if src_sr == dst_sr:
        return x
    if importlib.util.find_spec("scipy") is not None:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(src_sr, dst_sr)
        return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)
    n_out = int(round(len(x) * dst_sr / src_sr))
    t_old = np.linspace(0.0, 1.0, len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def _to_audio_dict(waveform: np.ndarray, sample_rate: int) -> dict:
    if torch is None:
        raise RuntimeError("torch 不可用，无法构造 AUDIO 数据（请确认在 ComfyUI 中运行）")
    return {
        "waveform": torch.from_numpy(np.asarray(waveform, dtype=np.float32))[None, None, :],
        "sample_rate": int(sample_rate),
    }


def _peak_norm(signal: np.ndarray) -> np.ndarray:
    """按峰值归一化到 0.5，与 modelscope pipeline 后处理保持一致。"""
    peak = float(np.abs(signal).max()) if signal.size else 0.0
    if peak > 1e-9:
        signal = signal / peak * 0.5
    return signal.astype(np.float32)


# --------------------------------------------------------------------------- #
# 后处理: 说话人活动门控（改善交替对话 / 非活跃段泄漏）
# --------------------------------------------------------------------------- #
def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """逐帧 RMS 能量（25ms 帧 / 10ms 步进，8000 Hz）。"""
    n = max(1, (len(x) - frame) // hop + 1)
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    idx = idx[idx.max(axis=1) < len(x)]
    seg = x[idx]
    return np.sqrt(np.mean(seg ** 2, axis=1)).astype(np.float32)


def _apply_gate(waveform: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    """说话人活动门控。

    - mode=off : 不处理
    - mode=soft: 按能量比例平滑压低非活跃段（不硬切，保留弱语音）
    - mode=hard: 低于阈值段静音（带 ~30ms 淡入淡出避免咔哒声）

    threshold 是相对该路峰值 0.5 的比例（gate_threshold 参数）。
    """
    if mode == "off" or len(waveform) == 0:
        return waveform
    fs = TARGET_SAMPLE_RATE
    frame = int(0.025 * fs)
    hop = int(0.010 * fs)
    rms = _frame_rms(waveform, frame, hop)
    peak = float(np.abs(waveform).max())
    thr = max(peak, 1e-9) * float(threshold)

    if mode == "hard":
        gain = (rms >= thr).astype(np.float32)
        kernel = np.ones(3) / 3.0                 # ~30ms 边缘平滑
    else:  # soft
        ratio = rms / (thr + 1e-12)
        gain = np.clip(ratio, 0.0, 1.0) ** 0.5    # 平方根: 平滑压低
        kernel = np.ones(5) / 5.0                 # ~50ms 时间平滑
    gain = np.convolve(gain, kernel, mode="same")

    frame_pos = np.arange(len(waveform)) // hop
    frame_pos = np.clip(frame_pos, 0, len(gain) - 1)
    return (waveform * gain[frame_pos]).astype(np.float32)


def _apply_mutual_gate(w1: np.ndarray, w2: np.ndarray, threshold: float,
                       strength_db: float = 20.0, full_db: float = 12.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """互斥门控（针对交替对话/非重叠语音）。

    原理：模型在交替对话上会把不活跃段的泄漏/串扰也保留在另一路。
    逐帧比较两路能量（dB 域），某路明显占优（该说话人正在说话）时压低另一路；
    两路能量接近（重叠说话或都静音）时两路都保留/都压低，不做误伤。

    - threshold : 占优死区（mutual_threshold，0~1）。|能量差占比| < threshold
      视为"同时说话/能量相当"，不抑制；越大越保守（0.25 ≈ ±4.4dB）。
    - strength_db : 最大抑制量（默认 20dB → 最弱通道最多压到 1/10）。
    - full_db     : 能量差达到该 dB 值即施加最大抑制（默认 12dB）。
    """
    if len(w1) == 0 or len(w2) == 0:
        return w1, w2
    fs = TARGET_SAMPLE_RATE
    frame = int(0.025 * fs)
    hop = int(0.010 * fs)
    r1 = _frame_rms(w1, frame, hop)
    r2 = _frame_rms(w2, frame, hop)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    if n == 0:
        return w1, w2

    eps = 1e-12
    log_r = 20.0 * np.log10((r1 + eps) / (r2 + eps))        # +: ch1 占优
    t = float(threshold)
    dz_db = 20.0 * np.log10((1.0 + t) / max(1e-6, 1.0 - t)) if t < 0.999 else 0.0
    mag = np.clip((np.abs(log_r) - dz_db) / max(1e-6, full_db - dz_db), 0.0, 1.0)
    g_weak = np.power(10.0, -strength_db * mag / 20.0)      # [1 → 0.1]
    g1 = np.where(log_r > 0, 1.0, g_weak)
    g2 = np.where(log_r < 0, 1.0, g_weak)

    both_quiet = np.maximum(r1, r2) < 1e-5
    g1 = np.where(both_quiet, 0.0, g1)
    g2 = np.where(both_quiet, 0.0, g2)

    # 非对称因果平滑：抑制（增益下降）快 ~25ms，恢复（增益上升）慢 ~60ms。
    # 泄漏在对方开口后立即被压下；说话人重新开口时增益温和回升，避免抽吸感。
    def _smooth_gain(g: np.ndarray) -> np.ndarray:
        out = np.empty_like(g)
        cur = float(g[0])
        a_down, a_up = 0.6, 0.35
        for i in range(len(g)):
            t = float(g[i])
            a = a_up if t > cur else a_down
            cur += a * (t - cur)
            out[i] = cur
        return out

    g1 = _smooth_gain(g1)
    g2 = _smooth_gain(g2)

    idx = np.arange(len(w1)) // hop
    idx = np.clip(idx, 0, n - 1)
    return (w1 * g1[idx]).astype(np.float32), (w2 * g2[idx]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 语音分析: 帧切分 / 基频 / 周期性 / 谱平坦度（用于固定 speaker_1 的输出排序）
# --------------------------------------------------------------------------- #
PITCH_MIN_HZ = 70.0
PITCH_MAX_HZ = 400.0
DEFAULT_GENDER_F0_THRESHOLD = 165.0


def _frame_stack(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """切帧 [n, frame]。"""
    n = max(1, (len(x) - frame) // hop + 1)
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    idx = idx[idx.max(axis=1) < len(x)]
    return x[idx]


def _frame_autocorr(frames: np.ndarray, max_lag: int) -> np.ndarray:
    """每帧自相关 r[0..max_lag]（FFT 快速实现，返回 [n, max_lag+1]）。"""
    nfft = 1
    while nfft < 2 * frames.shape[1]:
        nfft *= 2
    X = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(X * np.conj(X), n=nfft, axis=1)[:, :max_lag + 1]
    return ac


def _analyze_speech(x: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE,
                    gender_f0_threshold: float = DEFAULT_GENDER_F0_THRESHOLD) -> dict:
    """单路语音分析：质量分 + 性别（基频）估计，用于输出排序。

    返回 dict:
      quality     : 语音质量分 0~1（浊音占比 + 谐波清晰度，噪声/混合音明显更低）
      is_female   : None | True | False（浊音帧不足或性别不清晰时为 None）
      confidence  : 性别置信 0~1
      median_f0   : 浊音帧中位基频（Hz），无浊音帧为 0
      voiced_ratio: 浊音帧占比
    """
    x = np.asarray(x, dtype=np.float32)
    empty = {"quality": 0.0, "is_female": None, "confidence": 0.0,
             "median_f0": 0.0, "voiced_ratio": 0.0}
    if len(x) < 400:
        return empty
    frame = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    frames = _frame_stack(x, frame, hop)
    frames = frames * np.hanning(frame)[None, :]

    min_lag = max(2, int(sample_rate / PITCH_MAX_HZ))
    max_lag = int(sample_rate / PITCH_MIN_HZ)
    ac = _frame_autocorr(frames, max_lag)
    e = ac[:, 0]                                   # 帧能量（未归一化自相关 r0）
    r = ac / (e[:, None] + 1e-12)                  # 归一化自相关
    best_idx = np.argmax(r[:, min_lag:max_lag + 1], axis=1)
    best_lag = best_idx + min_lag
    periodicity = r[np.arange(len(frames)), best_lag]
    pitch = sample_rate / best_lag.astype(np.float64)

    # 谱平坦度: 谐波丰富的语音低、噪声高
    mag = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    flat = np.exp(np.mean(np.log(mag[:, 1:]), axis=1)) / (np.mean(mag[:, 1:], axis=1) + 1e-12)

    emax = float(e.max()) if e.size else 0.0
    voiced = (periodicity > 0.45) & (e > max(1e-6, emax * 1e-4))
    n_voiced = int(voiced.sum())

    voiced_ratio = n_voiced / len(frames) if len(frames) else 0.0
    clarity = float(np.mean(1.0 - np.clip(flat[voiced], 0.0, 1.0))) if n_voiced else 0.0
    quality = float(np.clip(0.6 * voiced_ratio + 0.4 * clarity, 0.0, 1.0))

    is_female = None
    confidence = 0.0
    median_f0 = 0.0
    if n_voiced >= 15:
        pv = pitch[voiced]
        median_f0 = float(np.median(pv))
        frac_f = float(np.mean(pv >= float(gender_f0_threshold)))
        confidence = float(np.clip(2.0 * abs(frac_f - 0.5), 0.0, 1.0))
        if confidence >= 0.35:
            is_female = bool(frac_f >= 0.5)
    return {"quality": quality, "is_female": is_female, "confidence": confidence,
            "median_f0": median_f0, "voiced_ratio": voiced_ratio}


def _gender_label(g: bool | None) -> str:
    return {True: "女", False: "男", None: "?"}.get(g, "?")


def _decide_order(m0: dict, m1: dict, mode: str) -> bool:
    """是否交换两路输出（返回 True=交换，使 speaker_1 变为原 spk2）。

    - as_is   : 不排序，保持模型原始输出
    - quality : speaker_1 = 质量分更高的那路
    - female  : speaker_1 = 女声（性别不确定时回退按质量）
    - auto    : 质量差距明显 → 按质量；质量接近且性别明确 → 女声优先 speaker_1
    """
    if mode == "as_is":
        return False
    q0, q1 = m0["quality"], m1["quality"]
    dq = q0 - q1
    f0, f1 = m0["is_female"], m1["is_female"]

    if mode == "quality":
        return bool(dq < 0)
    if mode == "female":
        if f0 is False and f1 is True:
            return True
        if f0 is True and f1 is False:
            return False
        return bool(dq < 0)
    # auto: 质量优先（保证 speaker_1 一定是较正常的人声），女声作次级
    if abs(dq) > 0.08:
        return bool(dq < 0)
    if f0 is False and f1 is True:
        return True
    if f0 is True and f1 is False:
        return False
    # 性别相同且质量接近（差 <0.02）时保持原序，避免无谓交换
    return bool(dq < -0.02)




# --------------------------------------------------------------------------- #
# 空洞修复（repair_mode=auto）
# --------------------------------------------------------------------------- #
def _detect_holes(spk1: np.ndarray, mix: np.ndarray,
                  sr: int = TARGET_SAMPLE_RATE, min_hole: float = 0.15) -> list:
    """检测 speaker_1 的静音空洞：spk1 帧 RMS 极低 且 输入混合该帧有语音。

    模型对弱音节/被吞内容会整体掩蔽，导致 speaker_1 该段静音而内容在
    speaker_2 或干脆丢失。返回 [(t0, t1), ...]（秒）。
    """
    if len(spk1) < 400 or len(mix) < 400:
        return []
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    r1 = _frame_rms(spk1, frame, hop)
    rm = _frame_rms(mix, frame, hop)
    n = min(len(r1), len(rm))
    hole = (r1[:n] < 0.005) & (rm[:n] > 0.012)
    segs, start = [], None
    for i in range(n):
        if hole[i] and start is None:
            start = i
        if not hole[i] and start is not None:
            if (i - start) * hop / sr >= min_hole:
                segs.append((start * hop / sr, i * hop / sr))
            start = None
    if start is not None and (n - start) * hop / sr >= min_hole:
        segs.append((start * hop / sr, n * hop / sr))
    return segs


def _speech_segments(mix: np.ndarray, sr: int = TARGET_SAMPLE_RATE,
                     gap: float = 0.35, min_len: float = 0.25) -> list:
    """把混合音频切成语音句：能量活动段，句内静音间隙 < gap 不切断。"""
    if len(mix) < 400:
        return []
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    r = _frame_rms(mix, frame, hop)
    thr = max(float(r.max()) * 0.03, 0.01)
    act = r > thr
    n = len(act)
    segs, start, last_end = [], None, -1
    for i in range(n):
        if act[i]:
            if start is None:
                start = i
            last_end = i
        elif start is not None and (i - last_end) * hop / sr > gap:
            if (last_end - start) * hop / sr >= min_len:
                segs.append((start * hop / sr, last_end * hop / sr))
            start = None
    if start is not None and (last_end - start) * hop / sr >= min_len:
        segs.append((start * hop / sr, last_end * hop / sr))
    return segs


def _repair_holes(mix: np.ndarray, spk1: np.ndarray, spk2: np.ndarray,
                  sr: int, infer_fn, min_hole: float = 0.15,
                  pad: float = 0.3) -> tuple:
    """空洞修复主流程（v3：音区一致性校验）。

    1) 检测 spk1 静音空洞。注意：两人轮流说话时，对方说话的时间段 spk1
       本就该静音——所以空洞只是"候选"，是否补入必须过第 5 步的音区校验；
    2) 空洞向两侧扩展（帧级阈值检测到的缺失区间往往更宽）；
    3) 以"空洞区间 ±pad"为窗口重新分离（窗口只含空洞附近内容，比整句
       更纯，模型掩蔽模式不同，弱音节往往能保住；也不会被同一句里的
       对方说话人内容污染）；
    4) 选路：空洞区间有内容的路优先（谁有缺失内容取谁）；
    5) 【关键】音区校验：补丁的基频与 spk1 全局基频差 > 50Hz → 判定为
       对方说话人（男声内容归 spk2），不补 spk1、也不清 spk2；
       基频为 0（清音弱音节）且质量合格 → 视为 spk1 被吞的弱音节，补；
    6) 通过校验的：增益匹配 + 交叉淡化补入 spk1，spk2 对应区间衰减 99%。

    infer_fn: 接收 1D np.ndarray，返回 (speaker_a, speaker_b) 两个 1D 数组。
    """
    if len(spk1) < 400 or len(mix) < 400:
        return spk1, spk2
    peak = float(np.abs(mix).max())
    mix_n = mix / peak if peak > 1e-9 else mix
    m_spk1 = _analyze_speech(spk1)
    spk1_f0 = float(m_spk1["median_f0"])

    holes = _detect_holes(spk1, mix_n, sr, min_hole)
    if not holes:
        return spk1, spk2
    # 空洞扩展 ±0.25s 并合并相邻空洞
    exp = 0.25
    holes = [(max(0.0, h0 - exp), min(len(mix_n) / sr, h1 + exp)) for h0, h1 in holes]
    merged = []
    for h0, h1 in sorted(holes):
        if merged and h0 <= merged[-1][1] + 0.4:
            merged[-1] = (merged[-1][0], max(merged[-1][1], h1))
        else:
            merged.append((h0, h1))
    holes = merged

    # 参考能量：spk1 语音活跃段（排除空洞区）
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    r1 = _frame_rms(spk1, frame, hop)
    active = np.zeros(len(spk1), dtype=bool)
    for i in range(len(r1)):
        a = i * hop
        b = min(a + frame, len(spk1))
        if r1[i] > 0.01:
            active[a:b] = True
    for h0, h1 in holes:
        active[max(int(h0 * sr), 0):int(h1 * sr)] = False
    ref = float(np.sqrt(np.mean(spk1[active] ** 2))) if active.any() else 0.08

    fade = int(0.1 * sr)
    out1 = spk1.copy().astype(np.float64)
    out2 = spk2.copy().astype(np.float64)

    for h0, h1 in holes:
        a = max(int((h0 - pad) * sr), 0)
        b = min(int((h1 + pad) * sr), len(mix_n))
        win = mix_n[a:b]
        if len(win) < 400:
            continue
        try:
            ra, rb = infer_fn(win)
        except Exception:
            continue
        lo = max(int(h0 * sr - a), 0)
        hi = min(int(h1 * sr - a), min(len(ra), len(rb)))
        if hi - lo < int(0.03 * sr):
            continue
        pa, pb = ra[lo:hi], rb[lo:hi]
        ea = float(np.sqrt(np.mean(pa ** 2))) if len(pa) else 0.0
        eb = float(np.sqrt(np.mean(pb ** 2))) if len(pb) else 0.0
        has_a, has_b = ea > 0.003, eb > 0.003
        if not has_a and not has_b:
            continue  # 两路该段都静音，补了也没用
        if has_a and not has_b:
            pick = pa
        elif has_b and not has_a:
            pick = pb
        else:
            # 两路都有内容：空洞区内容显著更多者优先（谁有缺失内容取谁）
            if eb > 3.0 * ea:
                pick = pb
            elif ea > 3.0 * eb:
                pick = pa
            else:
                ma = _analyze_speech(ra)
                mb = _analyze_speech(rb)
                pick = pa if ma["quality"] >= mb["quality"] else pb

        # 【关键】双重校验：补丁说话人与 spk1 是否同一人
        pm = _analyze_speech(pick)
        pf0 = float(pm["median_f0"])
        # ① 音区校验：补丁有基频且与 spk1 基频差大 → 对方说话人
        if pf0 > 0 and spk1_f0 > 0:
            _diff = abs(pf0 - spk1_f0)
            _ratio = _diff / max(spk1_f0, 1.0)
            if _diff > 50.0 and _ratio > 0.3:
                print(f"[FlatSepReformer] repair 跳过空洞 {h0:.2f}-{h1:.2f}s: "
                      f"补丁F0={pf0:.0f}Hz vs spk1={spk1_f0:.0f}Hz，音区不一致"
                      f"（视为对方说话人，保留在 spk2）")
                continue
        # ② 句内延续校验（无条件）：空洞必须是"spk1 句子中间缺失"——
        #    空洞紧邻后 spk1 仍有语音（句子没说完）；若后邻静音，说明空洞是
        #    句边界/孤立单字（如对方回应"好"），即使补丁音区接近也不能修
        #    （空洞扩展区常混入女声句尾，会骗过音区校验）。
        _post = spk1[int(h1 * sr + int(0.05 * sr)):int(h1 * sr + int(0.35 * sr))]
        _post_rms = float(np.sqrt(np.mean(_post ** 2))) if len(_post) else 0.0
        if _post_rms < 0.005:
            print(f"[FlatSepReformer] repair 跳过空洞 {h0:.2f}-{h1:.2f}s: "
                  f"句尾/孤立（后邻spk1 RMS={_post_rms:.4f}），"
                  f"视为对方说话人，保留在 spk2")
            continue
        if pf0 <= 0 and pm["quality"] < 0.4:
            print(f"[FlatSepReformer] repair 跳过空洞 {h0:.2f}-{h1:.2f}s: "
                  f"清音且质量低({pm['quality']:.2f})，不冒险补入")
            continue

        gain = min(max(ref / max(ea if pick is pa else eb, 1e-9), 0.5), 2.0)
        patch = pick * gain
        dst = int(h0 * sr)
        L = len(patch)
        for k in range(L):
            i0 = dst + k
            if i0 >= len(out1):
                break
            alpha = 1.0
            if k < fade:
                alpha = k / fade
            if L - k < fade:
                alpha = min(alpha, (L - k) / fade)
            out1[i0] = out1[i0] * (1 - alpha) + patch[k] * alpha
        # spk2 同步清理（空洞 ±0.15s，衰减 99%）
        c0 = max(int((h0 - 0.15) * sr), 0)
        c1 = min(int((h1 + 0.15) * sr), len(out2))
        Lc = c1 - c0
        for k in range(Lc):
            i0 = c0 + k
            alpha = 1.0
            if k < fade:
                alpha = k / fade
            if Lc - k < fade:
                alpha = min(alpha, (Lc - k) / fade)
            out2[i0] = out2[i0] * (1 - alpha * 0.99)

    out1 = np.clip(out1, -0.999, 0.999).astype(np.float32)
    out2 = np.clip(out2, -0.999, 0.999).astype(np.float32)
    return out1, out2


# --------------------------------------------------------------------------- #
# 目标说话人提取（target_speaker=female/male 的通用兜底）
# --------------------------------------------------------------------------- #
def _active_rms(x, sr, rel_thr=0.1, min_abs=0.003):
    """语音活跃段 RMS（帧 25ms/hop 10ms，按相对+绝对阈值）。"""
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    n = (len(x) - frame) // hop + 1
    if n < 1:
        return float(np.sqrt(np.mean(x ** 2)))
    r = np.array([np.sqrt(np.mean(x[i * hop:i * hop + frame] ** 2)) for i in range(n)])
    thr = max(r.max() * rel_thr, min_abs)
    act = r[r > thr]
    return float(act.mean()) if act.size else 0.0


def _split_utterances(x, sr, min_gap=0.30, rel_thr=0.05, min_abs=0.002):
    """按停顿切话语段。返回 [(start_idx, end_idx)]（帧索引，帧=32ms/hop16ms）。"""
    frame, hop = int(0.032 * sr), int(0.016 * sr)
    n = (len(x) - frame) // hop + 1
    if n < 4:
        return []
    rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + frame] ** 2)) for i in range(n)])
    thr = max(rms.max() * rel_thr, min_abs)
    act = rms > thr
    # 轻微膨胀活跃区（避免切在字尾）
    k = max(1, int(0.05 * sr) // hop)
    act = np.convolve(act, np.ones(k, dtype=np.int32), 'same') > 0
    utts = []
    start = None
    gap_frames = max(1, int(min_gap * sr) // hop)
    last_act = -1
    for i in range(n):
        if act[i]:
            if start is None:
                start = i
            last_act = i
        else:
            if start is not None and (i - last_act) > gap_frames:
                utts.append((start, last_act + 1))
                start = None
    if start is not None:
        utts.append((start, last_act + 1))
    # 过滤超短段
    return [u for u in utts if (u[1] - u[0]) * hop / sr >= 0.12]


def _utt_spectral_centroid(x, frame, hop, sr, rel_thr=0.15, min_abs=0.002):
    """话语段谱质心中位（Hz）。"""
    seg = x
    n = (len(seg) - frame) // hop + 1
    if n < 3:
        return None
    rms = np.array([np.sqrt(np.mean(seg[i * hop:i * hop + frame] ** 2)) for i in range(n)])
    thr = max(rms.max() * rel_thr, min_abs)
    idx = np.where(rms > thr)[0]
    if len(idx) < 4:
        return None
    cents = []
    for i in idx:
        w = seg[i * hop:i * hop + frame]
        if len(w) < frame:
            continue
        w = w * np.hanning(frame)
        sp = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        tot = np.sum(sp) + 1e-9
        cents.append(np.sum(freqs * sp) / tot)
    return float(np.median(cents)) if cents else None


def _kmeans2_split(vals):
    """2 簇分割：排序后找"最大且两边都有 >=2 个样本"的间隙，用间隙中点切分。

    返回 (low_center, high_center, low_mask, high_mask) 或 None。
    孤立点（只切出 1 个样本的间隙）会被过滤，避免单点伪峰。
    簇中心差 < 400 视为单一音色路，返回 None。
    """
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    if n < 5:
        return None
    sv = np.sort(vals)
    gaps = np.diff(sv)
    best = None
    for i, g in enumerate(gaps):
        cut = i + 1
        if cut < 2 or n - cut < 2:
            continue
        if best is None or g > best[0]:
            best = (g, cut)
    if best is None:
        return None
    _, cut = best
    low_c, high_c = float(np.median(sv[:cut])), float(np.median(sv[cut:]))
    if high_c - low_c < 400:
        return None
    mid = (sv[cut - 1] + sv[cut]) / 2.0
    low_mask = vals <= mid
    high_mask = ~low_mask
    if low_mask.sum() < 2 or high_mask.sum() < 2:
        return None
    return (low_c, high_c, low_mask, high_mask)


def _utt_f0_median(x, sr, frame, hop, lo=70, hi=500, rel_thr=0.12, min_abs=0.002):
    """话语段浊音帧基频中位（Hz），无浊音返回 None。"""
    from scipy.signal import find_peaks as _fp
    n = (len(x) - frame) // hop + 1
    if n < 4:
        return None
    rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + frame] ** 2)) for i in range(n)])
    thr = max(rms.max() * rel_thr, min_abs)
    lags = np.arange(int(sr / hi), int(sr / lo))
    f0s = []
    for i in range(n):
        if rms[i] <= thr:
            continue
        w = x[i * hop:i * hop + frame]
        if len(w) < frame:
            continue
        w = w - w.mean()
        ac = np.correlate(w, w, 'full')[len(w) - 1:]
        ac = ac / max(ac[0], 1e-9)
        s2 = ac[lags]
        if len(s2) < 10:
            continue
        pk, props = _fp(s2, height=0.3)
        if len(pk) == 0:
            continue
        f0 = sr / lags[pk[np.argmax(props['peak_heights'])]]
        if lo <= f0 <= hi:
            f0s.append(f0)
    return float(np.median(f0s)) if len(f0s) >= 4 else None


def _detect_mixed_route(w0, w1, sr):
    """检测两路中是否存在'混合语音路'（一路里男女话语都有、另一路是音乐/残渣）。

    返回 (voice_index, low_utts, high_utts, (low_center, high_center)) 或 None。
    判定：话语段谱质心 2-means 双簇（簇差>400 且每簇>=2 段）且高簇质心>1150
    （明显女声音色）且低簇基频中位<200（明显男声）。两路都是正常语音时
    （各自单簇，或高簇不够女声）返回 None，回退正常分离逻辑。
    """
    frame, hop = int(0.032 * sr), int(0.016 * sr)
    for vi, x in enumerate((w0, w1)):
        utts = _split_utterances(x, sr)
        if len(utts) < 5:
            continue
        feats = []
        for a, b in utts:
            seg = x[a * hop:min(b * hop, len(x))]
            c = _utt_spectral_centroid(seg, frame, hop, sr)
            if c is not None:
                feats.append(c)
        if len(feats) < 4:
            continue
        r = _kmeans2_split(feats)
        if r is None:
            continue
        low_c, high_c, low_mask, high_mask = r
        if high_c - low_c < 400 or high_c < 1150:
            continue
        # ---- 边界修正：质心落在两簇中点 ±15% 的话语段用基频裁决 ----
        # 女声低音/快语速段的谱质心可能贴近男声簇，男声高音段反之；
        # 谱质心模糊时基频（F0>200 女声 / <160 男声）更可靠。
        mid = (low_c + high_c) / 2.0
        for i, c in enumerate(feats):
            if mid * 0.85 <= c <= mid * 1.15:
                a, b = utts[i]
                seg = x[a * hop:min(b * hop, len(x))]
                f = _utt_f0_median(seg, sr, frame, hop)
                if f is None:
                    continue
                if f > 200:
                    low_mask[i], high_mask[i] = False, True
                elif f < 160:
                    low_mask[i], high_mask[i] = True, False
        low_utts = [utts[i] for i in np.where(low_mask)[0]]
        high_utts = [utts[i] for i in np.where(high_mask)[0]]
        if len(low_utts) < 1 or len(high_utts) < 1:
            continue
        # 低簇（男声）基频中位必须明显低于女声阈值
        low_f0s = []
        for a, b in low_utts:
            seg = x[a * hop:min(b * hop, len(x))]
            f = _utt_f0_median(seg, sr, frame, hop)
            if f is not None:
                low_f0s.append(f)
        if not low_f0s or np.median(low_f0s) >= 200:
            continue
        return (vi, low_utts, high_utts, (low_c, high_c))
    return None


def _concat_utterances(x, mask, sr):
    """按话语段掩码把 x 对应时间段拼到时间轴上（保留静音间隔）。"""
    out = np.zeros_like(x)
    frame, hop = int(0.032 * sr), int(0.016 * sr)
    for a, b in mask:
        out[a * hop:min(b * hop, len(x))] = x[a * hop:min(b * hop, len(x))]
    return out



def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _pick_backend(model_dir: str, backend: str) -> str:
    """backend=auto 时优先 onnx（无需 modelscope），否则回退 modelscope。"""
    if backend != "auto":
        return backend
    if os.path.isfile(os.path.join(model_dir, "onnx_model.onnx")) and \
            importlib.util.find_spec("onnxruntime") is not None:
        return "onnx"
    return "modelscope"


def _load_onnx_model(model_dir: str, device: str) -> dict:
    onnx_path = os.path.join(model_dir, "onnx_model.onnx")
    if not os.path.isfile(onnx_path):
        raise RuntimeError(f"未找到 ONNX 模型文件: {onnx_path}")
    try:
        import onnxruntime
    except ImportError:
        raise RuntimeError(
            "缺少 onnxruntime 依赖，请执行: pip install onnxruntime "
            "（或使用 modelscope 后端）")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device != "cpu" else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return {"backend": "onnx", "session": session, "dir": model_dir}


def _load_modelscope_model(model_dir: str, device: str) -> dict:
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except ImportError:
        raise RuntimeError(
            "缺少 modelscope 依赖，请执行: pip install -r requirements.txt")
    try:
        pipe = pipeline(Tasks.speech_separation, model=model_dir, device=device)
    except KeyError as e:
        raise RuntimeError(
            "当前 modelscope 版本无法识别该模型（registry 中缺少 "
            "speech_flatsepreformer_separation_temporal_8k_base_libri2mix100）。\n"
            "解决办法（任选其一）:\n"
            "  1. 推荐: 节点 backend 选择 onnx（使用模型自带的 onnx_model.onnx，"
            "无需 modelscope）\n"
            "  2. 升级 modelscope 到 master 源码版:\n"
            "     pip install -U \"modelscope @ "
            "git+https://github.com/modelscope/modelscope.git@master\""
        ) from e
    return {"backend": "modelscope", "pipeline": pipe, "dir": model_dir}


def _load_model(model_dir: str, backend: str, device: str) -> dict:
    backend = _pick_backend(model_dir, backend)
    dev = _resolve_device(device)
    key = (model_dir, backend, dev)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = (_load_onnx_model(model_dir, dev) if backend == "onnx"
             else _load_modelscope_model(model_dir, dev))
    _MODEL_CACHE[key] = model
    return model


def _infer_onnx(session, waveform: np.ndarray) -> list[np.ndarray]:
    """ONNX Runtime 推理: 输入 mixture [1, T] float32，输出 [spk0, spk1]。"""
    inp = waveform[None].astype(np.float32)  # [1, T]
    out = session.run(None, {"mixture": inp})[0]
    arr = np.asarray(out)
    if arr.ndim == 3:
        arr = arr[0]  # [num_spk, T] 或 [T, num_spk]
    if arr.ndim != 2:
        raise RuntimeError(f"ONNX 输出维度异常: {arr.shape}")
    # 归一化为 [num_spk, T]
    if arr.shape[0] == NUM_SPEAKERS and arr.shape[1] != NUM_SPEAKERS:
        pass
    elif arr.shape[1] == NUM_SPEAKERS:
        arr = arr.T
    else:
        raise RuntimeError(f"ONNX 输出无法解析为 {NUM_SPEAKERS} 路: {arr.shape}")
    # 与 modelscope 后端一致: 按峰值归一化到 0.5
    return [_peak_norm(arr[0]), _peak_norm(arr[1])]


def _infer_modelscope(pipeline_obj, waveform: np.ndarray) -> list[np.ndarray]:
    """ModelScope pipeline 推理: 输入 8k mono float32，返回 [spk0, spk1]。"""
    if sf is None:
        raise RuntimeError("缺少 soundfile 依赖，请执行: pip install -r requirements.txt")
    from modelscope.outputs import OutputKeys

    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="flatsep_")
        os.close(fd)
        sf.write(tmp_path, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")
        result = pipeline_obj(tmp_path)
        pcm_list = result[OutputKeys.OUTPUT_PCM_LIST]
        if not pcm_list or len(pcm_list) < NUM_SPEAKERS:
            raise RuntimeError(f"模型输出异常，期望 {NUM_SPEAKERS} 路，实际 {len(pcm_list or [])} 路")
        spks = []
        for b in pcm_list[:NUM_SPEAKERS]:
            arr = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
            spks.append(arr)
        return spks
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# 合并节点: 模型加载 + 语音分离（v2）
# --------------------------------------------------------------------------- #
class FlatSepReformerSeparate:
    """双说话人语音分离（自动加载模型 + 可调后处理）。

    - 模型自动从 <ComfyUI>/models/FlatSepReformer 加载（相对插件路径推导）。
    - backend=auto 时优先使用 ONNX（无需 modelscope）。
    - 输出排序 output_order（解决"哪路是 speaker_1 不固定"）:
      * auto     : speaker_1 = 质量更好的那路；质量接近时优先女声（默认）
      * quality  : speaker_1 = 质量更好的那路
      * female   : speaker_1 = 女声（性别不明时回退按质量）
      * as_is    : 保持模型原始输出顺序（旧行为）
    - 后处理参数（按场景手动调节）:
      * gate_mode : off/soft/hard + mutual（互斥门控，专治交替对话串扰）
      * mutual_threshold : 互斥门控的占优死区
      * gate_threshold   : soft/hard 门控阈值
      * gender_f0_threshold : 女声判定基频阈值（Hz）
      * output_gain / match_input_sr : 音量与采样率
    """

    DESCRIPTION = (
        "【双说话人语音分离】\n"
        "把一段两人混合语音分离为两路独立人声（自动加载 "
        "<ComfyUI>/models/FlatSepReformer 模型，无需填路径）。\n\n"
        "【输出排序 output_order · 解决 speaker_1/speaker_2 不固定】\n"
        "模型是置换不变训练，原始输出哪路对应谁不固定。本节点默认 auto：\n"
        "  1) 先按语音质量排序，质量好的固定为 speaker_1（不会出现\n"
        "     speaker_1 是杂音的情况）\n"
        "  2) 两路质量接近时，女声固定输出到 speaker_1\n"
        "  选项：auto=质量优先+女声次级（默认）/ quality=仅按质量 /\n"
        "        female=固定女声在 speaker_1 / as_is=保持原始顺序\n\n"
        "【gate_mode 门控 · 解决交替对话串扰】\n"
        "两人轮流说话（如影视对白）时，模型会把不活跃段的泄漏也保留：\n"
        "  · off   ：不处理，输出模型原始结果（默认）\n"
        "  · soft  ：按能量平滑压低非活跃段，轻微夹杂用\n"
        "  · hard  ：低于阈值直接静音，夹杂严重时用\n"
        "  · mutual：互斥门控（推荐用于交替对话）——逐帧比较两路能量，\n"
        "    某路明显占优时压低另一路泄漏，两路都更干净；\n"
        "    重叠说话/静音段不误伤，配合 mutual_threshold 调节灵敏度\n\n"
        "【其他参数】\n"
        "mutual_threshold（互斥门控死区，默认0.25，越大越保守）、\n"
        "gate_threshold（soft/hard 阈值，建议0.02起调）、\n"
        "gender_f0_threshold（女声基频判定，默认165Hz）、\n"
        "output_gain（输出增益0.1~4.0）、\n"
        "match_input_sr（输出采样率匹配输入，避免8k听感偏闷）。\n\n"
        "【能力边界】模型为 8kHz 干净双说话人分离，不含去噪/去背景音；\n"
        "背景音需先用 UVR、Demucs 等专用工具去除后再送入本节点。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "backend": (
                    ["auto", "onnx", "modelscope"],
                    {"default": "auto",
                     "tooltip": "推理后端：auto=优先 onnx（无需 modelscope，推荐）；"
                                "onnx=使用 onnx_model.onnx；"
                                "modelscope=使用 pytorch_model.pt（需 master 版 modelscope）"},
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto", "tooltip": "计算设备：auto=有 CUDA 用 GPU，否则 CPU"},
                ),
                "gate_mode": (
                    ["off", "soft", "hard", "mutual"],
                    {"default": "off",
                     "tooltip": "门控模式：off=不处理；soft=平滑压低非活跃段；"
                                "hard=低于阈值直接静音；"
                                "mutual=互斥门控（推荐用于两人轮流说话的交替对话，"
                                "逐帧比较两路能量压低另一路泄漏）"},
                ),
                "gate_threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001,
                     "tooltip": "soft/hard 门控阈值（相对该路峰值 0.5 的比例）。"
                                "越小越灵敏，容易把轻语音也压掉；越大抑制越强。"
                                "建议 0.02 起调，夹杂仍重就逐步加大到 0.05~0.1"},
                ),
                "output_gain": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                     "tooltip": "输出增益：分离后音量偏小就调大（如 1.5~2.0），"
                                "偏大就调小；超过 1.0 的部分会被截断（削波）"},
                ),
                "match_input_sr": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "输出采样率匹配输入音频。模型固定输出 8000Hz"
                                "（高频截止 4kHz，听感偏闷）；输入是 44.1k/48k 等时"
                                "开启可让输出回到原采样率，听感更清晰；关闭则输出固定 8000Hz"},
                ),
                "output_order": (
                    ["auto", "quality", "female", "as_is"],
                    {"default": "auto",
                     "tooltip": "输出排序（固定 speaker_1）：auto=质量优先+女声次级（推荐，"
                                "保证 speaker_1 一定是较正常的人声，质量接近时女声固定到 speaker_1）；"
                                "quality=仅按质量排序；female=女声固定到 speaker_1；"
                                "as_is=保持模型原始顺序"},
                ),
                "mutual_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.05, "max": 0.9, "step": 0.01,
                     "tooltip": "互斥门控占优死区（0~1）：两路能量差占比小于该值视为"
                                "同时说话/静音不抑制。越小抑制越激进（0.1~0.15 压制串扰更强），"
                                "越大越保守（0.4+ 只处理极明显占优段）"},
                ),
                "gender_f0_threshold": (
                    "FLOAT",
                    {"default": DEFAULT_GENDER_F0_THRESHOLD, "min": 100.0, "max": 300.0,
                     "step": 5.0,
                     "tooltip": "女声判定基频阈值（Hz）：浊音帧中位基频 ≥ 该值判为女声。"
                                "男声通常 70~180Hz，女声通常 165~350Hz。"
                                "低音女声可调到 150，高音男声可调到 180"},
                ),
                "repair_mode": (
                    ["off", "auto"],
                    {"default": "off",
                     "tooltip": "空洞修复（解决 speaker_1 丢字/缺内容）：auto=检测 speaker_1 "
                                "静音但输入有语音的段，用该句窗口重新分离后把被模型吞掉的内容 "
                                "补回 speaker_1，并同步清理 speaker_2 里的重复女声。"
                                "off=不修复（默认）。模型对影视对白的弱音节（如唇、破等）"
                                "常直接掩蔽丢弃，此模式可在后处理阶段找回"},
                ),
                "repair_min_hole": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.05, "max": 2.0, "step": 0.01,
                     "tooltip": "空洞修复的最小空洞时长（秒）：speaker_1 静音且输入有语音的段"
                                "短于该值不修复。默认 0.15"},
                ),
                "repair_pad": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 3.0, "step": 0.05,
                     "tooltip": "空洞修复的重分离窗口上下文余量（秒）：重分离时在句子前后各"
                                "多取的音频，给模型更多上下文，分离更稳。默认 0.3"},
                ),
                "target_speaker": (
                    ["auto", "female", "male"],
                    {"default": "auto",
                     "tooltip": "目标说话人（通用兜底）：auto=正常双路分离。当输入含背景音乐"
                                "或两人音色接近导致模型无法分开时，选 female/male 启用"
                                "‘目标说话人提取’：自动检测混合语音路，按话语段的谱质心"
                                "聚类（男低女高），把目标性别的完整话语段输出到 speaker_1，"
                                "另一人输出到 speaker_2。检测不到混合语音路时自动回退"
                                "正常分离逻辑，不影响原结果"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("speaker_1", "speaker_2")
    FUNCTION = "separate"
    CATEGORY = "audio/separation"

    def separate(self, audio: dict, backend: str = "auto", device: str = "auto",
                 gate_mode: str = "off", gate_threshold: float = 0.02,
                 output_gain: float = 1.0, match_input_sr: bool = False,
                 output_order: str = "auto", mutual_threshold: float = 0.25,
                 gender_f0_threshold: float = DEFAULT_GENDER_F0_THRESHOLD,
                 repair_mode: str = "off", repair_min_hole: float = 0.15,
                 repair_pad: float = 0.3, target_speaker: str = "auto"):
        model_dir = find_model_dir()
        if not model_dir:
            # 运行时自动下载到默认目标文件夹: <ComfyUI>/models/FlatSepReformer
            target = _default_model_dir()
            _auto_download_model(target)
            model_dir = find_model_dir(target)
        if not model_dir:
            raise RuntimeError(
                "未找到模型目录。请确认模型已放入 <ComfyUI>/models/FlatSepReformer/ "
                "（运行节点会自动下载；也可用夸克网盘或魔塔下载后放入），"
                "或设置环境变量 FLATSEPREFORMER_MODEL_DIR。")

        model = _load_model(model_dir, backend, device)

        waveform, input_sr = _to_mono_float32(audio)
        if len(waveform) == 0:
            raise ValueError("输入音频为空")
        if input_sr != TARGET_SAMPLE_RATE:
            waveform = _resample(waveform, input_sr, TARGET_SAMPLE_RATE)

        spks = (_infer_onnx(model["session"], waveform)
                if model["backend"] == "onnx"
                else _infer_modelscope(model["pipeline"], waveform))

        # ---- 目标说话人提取（target_speaker=female/male）----
        # 模型在"输入含背景音乐/哼唱"或"两人音色接近"时可能把两个真人语音挤到
        # 同一路、另一路变成音乐/残渣。此时切换到"语音路 utterance 级谱质心聚类"：
        # 对语音路按停顿切成话语段，每段谱质心中位聚类（男低女高），把目标性别的
        # 话语段完整提取出来。若两路都是正常语音（未检测到混合语音路），回退到
        # 下方原有 gate/排序/repair 逻辑，旧工作流行为完全不变。
        if target_speaker in ("female", "male"):
            route = _detect_mixed_route(spks[0], spks[1], TARGET_SAMPLE_RATE)
            if route is not None:
                vi, low_mask, high_mask, feats = route
                voice = spks[vi]
                is_female = target_speaker == "female"
                sel = high_mask if is_female else low_mask
                other = low_mask if is_female else high_mask
                spk1 = _concat_utterances(voice, sel, TARGET_SAMPLE_RATE)
                spk2 = _concat_utterances(voice, other, TARGET_SAMPLE_RATE)
                # 增益对齐 spk1 活跃段 RMS，限幅防削波
                ref = _active_rms(spk1, TARGET_SAMPLE_RATE)
                if ref > 1e-4:
                    for k, s in enumerate((spk1, spk2)):
                        if _active_rms(s, TARGET_SAMPLE_RATE) > 1e-4:
                            g = np.clip(ref / _active_rms(s, TARGET_SAMPLE_RATE), 0.5, 2.0)
                            spks[k] = np.clip(s * g, -1.0, 1.0)
                        else:
                            spks[k] = s
                else:
                    spks = [spk1, spk2]
                print(
                    f"[FlatSepReformer] target_speaker={target_speaker} 检测到混合语音路"
                    f"（模型未把两人分开），按话语段谱质心聚类提取："
                    f"质心低簇={feats[0]:.0f}Hz 高簇={feats[1]:.0f}Hz")
                # 提取模式下 spk1 已完整，跳过 gate/排序/repair
                return self._finalize(
                    spks, audio, waveform, output_gain, match_input_sr)

        # ---- 后处理（可调参数）----
        if gate_mode == "mutual":
            w0, w1 = _apply_mutual_gate(spks[0], spks[1], mutual_threshold)
            spks = [w0, w1]
        elif gate_mode != "off":
            spks = [_apply_gate(s, gate_mode, gate_threshold) for s in spks]
        spks = [_peak_norm(s) for s in spks]

        # ---- 输出排序：固定 speaker_1（质量优先，女声次级）----
        swapped = False
        if output_order != "as_is":
            m0 = _analyze_speech(spks[0], TARGET_SAMPLE_RATE, gender_f0_threshold)
            m1 = _analyze_speech(spks[1], TARGET_SAMPLE_RATE, gender_f0_threshold)
            swapped = _decide_order(m0, m1, output_order)
            if swapped:
                spks.reverse()
                m0, m1 = m1, m0
            print(
                f"[FlatSepReformer] output_order={output_order} 交换={swapped} | "
                f"质量 spk1={m0['quality']:.3f} spk2={m1['quality']:.3f} | "
                f"F0 spk1={m0['median_f0']:.0f}Hz spk2={m1['median_f0']:.0f}Hz | "
                f"性别 spk1={_gender_label(m0['is_female'])} "
                f"spk2={_gender_label(m1['is_female'])}")

        # ---- 空洞修复：spk1 静音但输入有语音的段，从混合句窗口重分离补回 ----
        if repair_mode == "auto":
            if model["backend"] == "onnx":
                def _infer_fn(win):
                    _out = _infer_onnx(model["session"], win)
                    return _out[0], _out[1]
            else:
                def _infer_fn(win):
                    _out = _infer_modelscope(model["pipeline"], win)
                    return _out[0], _out[1]
            spks = list(_repair_holes(
                waveform, spks[0], spks[1], TARGET_SAMPLE_RATE,
                _infer_fn, repair_min_hole, repair_pad))
            print(f"[FlatSepReformer] repair_mode=auto 空洞修复完成")

        # ---- 增益 + 输出采样率 ----
        return self._finalize(spks, audio, waveform, output_gain, match_input_sr)

    def _finalize(self, spks, audio, waveform, output_gain, match_input_sr):
        """增益 + 输出采样率（供正常流程与目标说话人提取共用）。"""
        input_sr = audio["sample_rate"]
        outs = []
        for spk in spks:
            if output_gain != 1.0:
                spk = np.clip(spk * float(output_gain), -1.0, 1.0)
            out_sr = TARGET_SAMPLE_RATE
            if match_input_sr and input_sr != TARGET_SAMPLE_RATE:
                spk = _resample(spk, TARGET_SAMPLE_RATE, input_sr)
                out_sr = input_sr
            outs.append(_to_audio_dict(spk, out_sr))

        return (outs[0], outs[1])


# --------------------------------------------------------------------------- #
# 音频加载 / 保存
# --------------------------------------------------------------------------- #
class FlatSepReformerLoadAudio:
    """从本地文件加载音频（wav/flac/ogg/mp3 等 soundfile 支持的格式）。"""

    DESCRIPTION = (
        "【加载音频】\n"
        "从本地文件读取音频（wav/flac/ogg/mp3 等 soundfile 支持的格式），\n"
        "输出标准 AUDIO（waveform + sample_rate），可直接连接分离节点。\n"
        "路径填音频文件的绝对路径；采样率任意，分离节点会自动重采样到 8000Hz。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": (
                    "STRING",
                    {"default": "", "multiline": False, "placeholder": "音频文件绝对路径"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = "audio/separation"

    def load(self, audio_path: str):
        if sf is None:
            raise RuntimeError("缺少 soundfile 依赖，请执行: pip install -r requirements.txt")
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = data[:, 0]
        return (_to_audio_dict(mono, int(sr)),)


class FlatSepReformerSaveAudio:
    """将 AUDIO 保存为 wav 文件，返回保存路径。"""

    DESCRIPTION = (
        "【保存音频】\n"
        "把 AUDIO 保存为 wav 文件，输出保存路径。\n"
        "filename 填文件名（不含扩展名，自动加 .wav）；\n"
        "output_dir 留空则保存到 <ComfyUI>/output/ 目录。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "separated_audio", "multiline": False}),
            },
            "optional": {
                "output_dir": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "placeholder": "留空使用 <ComfyUI>/output"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "audio/separation"

    def save(self, audio: dict, filename: str, output_dir: str = ""):
        if sf is None:
            raise RuntimeError("缺少 soundfile 依赖，请执行: pip install -r requirements.txt")
        waveform, sr = _to_mono_float32(audio)
        out_dir = output_dir.strip() or _default_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        safe = "".join(c for c in filename if c not in '\\/:*?"<>|').strip() or "separated_audio"
        path = os.path.join(out_dir, f"{safe}.wav")
        sf.write(path, waveform, sr, subtype="PCM_16")
        return (path,)


# --------------------------------------------------------------------------- #
# 注册
# --------------------------------------------------------------------------- #
NODE_CLASS_MAPPINGS = {
    "FlatSepReformerSeparate": FlatSepReformerSeparate,
    "FlatSepReformerLoadAudio": FlatSepReformerLoadAudio,
    "FlatSepReformerSaveAudio": FlatSepReformerSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlatSepReformerSeparate": "FlatSepReformer (Separate 2 Speakers)",
    "FlatSepReformerLoadAudio": "Load Audio (FlatSepReformer)",
    "FlatSepReformerSaveAudio": "Save Audio (FlatSepReformer)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

