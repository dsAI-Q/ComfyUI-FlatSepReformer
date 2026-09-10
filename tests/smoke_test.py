# -*- coding: utf-8 -*-
"""
端到端冒烟测试：直接调用节点类，模拟 ComfyUI 节点执行链路。

用法:
    cd ComfyUI-FlatSepReformer
    python tests/smoke_test.py

前置:
    - 模型已下载到 <repo>/models/FlatSepReformer/（或使用 install.py）
    - 已安装依赖: pip install -r requirements.txt
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nodes  # noqa: E402

AUDIO_PATH = os.path.join(ROOT, "examples", "mix_speech1.wav")
OUT_DIR = os.path.join(ROOT, "output")


def main() -> None:
    # [1] LoadAudio
    loader = nodes.FlatSepReformerLoadAudio()
    (audio,) = loader.load(AUDIO_PATH)
    assert audio["waveform"].ndim == 3, audio["waveform"].shape
    assert audio["sample_rate"] > 0
    print(f"[1] LoadAudio OK: sr={audio['sample_rate']} "
          f"shape={tuple(audio['waveform'].shape)}")

    # [2] Loader (onnx backend)
    model_ld = nodes.FlatSepReformerLoader()
    (model_onnx,) = model_ld.load("", "onnx", "cpu")
    assert model_onnx["backend"] == "onnx"
    print(f"[2] Loader(onnx) OK: dir={model_onnx['dir']}")

    # [3] Separate (onnx)
    sep = nodes.FlatSepReformerSeparate()
    s1, s2 = sep.separate(model_onnx, audio)
    w1, w2 = s1["waveform"][0, 0].numpy(), s2["waveform"][0, 0].numpy()
    assert s1["sample_rate"] == 8000 and s2["sample_rate"] == 8000
    assert w1.shape == w2.shape, (w1.shape, w2.shape)
    assert float(np.abs(w1).max()) > 0.01 and float(np.abs(w2).max()) > 0.01
    print(f"[3] Separate(onnx) OK: len={w1.shape[0]} sr=8000 "
          f"max1={np.abs(w1).max():.3f} max2={np.abs(w2).max():.3f}")

    # [4] SaveAudio
    saver = nodes.FlatSepReformerSaveAudio()
    (p1,) = saver.save(s1, "spk1_onnx", OUT_DIR)
    (p2,) = saver.save(s2, "spk2_onnx", OUT_DIR)
    assert os.path.isfile(p1) and os.path.isfile(p2)
    print(f"[4] SaveAudio OK:\n    {p1}\n    {p2}")

    # [5] Loader + Separate (modelscope backend)
    (model_ms,) = model_ld.load("", "modelscope", "cpu")
    assert model_ms["backend"] == "modelscope"
    ms1, ms2 = sep.separate(model_ms, audio)
    wm1, wm2 = ms1["waveform"][0, 0].numpy(), ms2["waveform"][0, 0].numpy()
    assert wm1.shape == wm2.shape
    assert float(np.abs(wm1).max()) > 0.01 and float(np.abs(wm2).max()) > 0.01
    print(f"[5] Separate(modelscope) OK: len={wm1.shape[0]} "
          f"max1={np.abs(wm1).max():.3f} max2={np.abs(wm2).max():.3f}")

    # [6] 非 8k 输入自动重采样（用 44100 Hz 模拟）
    fake_sr = 44100
    fake_len = int(len(w1) * fake_sr / 8000)
    fake = np.sin(2 * np.pi * 220 * np.arange(fake_len) / fake_sr).astype(np.float32)
    import torch
    fake_audio = {"waveform": torch.from_numpy(np.asarray(fake, dtype=np.float32))[None, None, :],
                  "sample_rate": fake_sr}
    r1, r2 = sep.separate(model_onnx, fake_audio)
    assert r1["sample_rate"] == 8000
    print(f"[6] Resample OK: {fake_sr}Hz -> 8000Hz, out_len={r1['waveform'].shape[-1]}")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
