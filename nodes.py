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
# 推理后端
# --------------------------------------------------------------------------- #
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
    """双说话人语音分离（自动加载模型）。

    - 模型自动从 <ComfyUI>/models/FlatSepReformer 加载（相对插件路径推导），
      无需单独加载节点、无需填绝对路径。
    - backend=auto 时优先使用 ONNX（无需 modelscope），
      modelscope 仅作为可选项。
    - 输入音频自动重采样到 8000 Hz。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "backend": (
                    ["auto", "onnx", "modelscope"],
                    {"default": "auto",
                     "tooltip": "auto: 优先 onnx（无需 modelscope，推荐）；"
                                "onnx: 使用 onnx_model.onnx；"
                                "modelscope: 使用 pytorch_model.pt（需 master 版 modelscope）"},
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto", "tooltip": "auto: 有 CUDA 用 GPU，否则 CPU"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("speaker_1", "speaker_2")
    FUNCTION = "separate"
    CATEGORY = "audio/separation"

    def separate(self, audio: dict, backend: str = "auto", device: str = "auto"):
        model_dir = find_model_dir()
        if not model_dir:
            raise RuntimeError(
                "未找到模型目录。请将模型放到 <ComfyUI>/models/FlatSepReformer/ "
                "（运行 python install.py 自动下载），"
                "或设置环境变量 FLATSEPREFORMER_MODEL_DIR。")

        model = _load_model(model_dir, backend, device)

        waveform, sr = _to_mono_float32(audio)
        if len(waveform) == 0:
            raise ValueError("输入音频为空")
        if sr != TARGET_SAMPLE_RATE:
            waveform = _resample(waveform, sr, TARGET_SAMPLE_RATE)

        spks = (_infer_onnx(model["session"], waveform)
                if model["backend"] == "onnx"
                else _infer_modelscope(model["pipeline"], waveform))

        return (
            _to_audio_dict(spks[0], TARGET_SAMPLE_RATE),
            _to_audio_dict(spks[1], TARGET_SAMPLE_RATE),
        )


# --------------------------------------------------------------------------- #
# 音频加载 / 保存
# --------------------------------------------------------------------------- #
class FlatSepReformerLoadAudio:
    """从本地文件加载音频（wav/flac/ogg/mp3 等 soundfile 支持的格式）。"""

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
