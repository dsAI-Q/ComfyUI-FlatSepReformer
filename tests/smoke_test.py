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
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nodes  # noqa: E402

AUDIO_PATH = os.path.join(ROOT, "examples", "mix_speech1.wav")
OUT_DIR = os.path.join(ROOT, "output")


def test_auto_does_not_import_modelscope() -> None:
    """验证 backend=auto 走 onnx 时不依赖 modelscope（规避 registry 报错）。"""
    code = (
        "import os, sys; sys.path.insert(0, %r); import nodes;"
        "m = nodes._load_model(nodes.find_model_dir(), 'auto', 'cpu');"
        "assert m['backend'] == 'onnx', m['backend'];"
        "assert 'modelscope' not in sys.modules, 'modelscope imported!';"
        "print('AUTO-BACKEND:', m['backend']); print('NO-MODELSCOPE: OK')"
    ) % ROOT
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=ROOT)
    out = (r.stdout + r.stderr).strip()
    assert r.returncode == 0, out
    assert "AUTO-BACKEND: onnx" in out and "NO-MODELSCOPE: OK" in out, out
    print("[0] auto->onnx 不依赖 modelscope OK")


def main() -> None:
    # [1] LoadAudio
    loader = nodes.FlatSepReformerLoadAudio()
    (audio,) = loader.load(AUDIO_PATH)
    assert audio["waveform"].ndim == 3, audio["waveform"].shape
    assert audio["sample_rate"] > 0
    print(f"[1] LoadAudio OK: sr={audio['sample_rate']} "
          f"shape={tuple(audio['waveform'].shape)}")

    # [2] 合并节点: backend=auto（应解析为 onnx）；as_is 保持原始顺序便于后端对比
    sep = nodes.FlatSepReformerSeparate()
    s1, s2 = sep.separate(audio, "auto", "cpu", output_order="as_is")
    w1, w2 = s1["waveform"][0, 0].numpy(), s2["waveform"][0, 0].numpy()
    assert s1["sample_rate"] == 8000 and s2["sample_rate"] == 8000
    assert w1.shape == w2.shape, (w1.shape, w2.shape)
    assert float(np.abs(w1).max()) > 0.01 and float(np.abs(w2).max()) > 0.01
    print(f"[2] Separate(auto->onnx) OK: len={w1.shape[0]} "
          f"max1={np.abs(w1).max():.3f} max2={np.abs(w2).max():.3f}")

    # [3] 模型路径自动探测（相对路径）
    md = nodes.find_model_dir()
    assert md and os.path.isdir(md), md
    print(f"[3] 模型自动探测 OK: {md}")

    # [4] SaveAudio
    saver = nodes.FlatSepReformerSaveAudio()
    (p1,) = saver.save(s1, "spk1_auto", OUT_DIR)
    (p2,) = saver.save(s2, "spk2_auto", OUT_DIR)
    assert os.path.isfile(p1) and os.path.isfile(p2)
    print(f"[4] SaveAudio OK:\n    {p1}\n    {p2}")

    # [5] backend=modelscope
    ms1, ms2 = sep.separate(audio, "modelscope", "cpu", output_order="as_is")
    wm1, wm2 = ms1["waveform"][0, 0].numpy(), ms2["waveform"][0, 0].numpy()
    assert wm1.shape == wm2.shape
    assert float(np.abs(wm1).max()) > 0.01 and float(np.abs(wm2).max()) > 0.01
    assert float(np.abs(wm1 - w1).max()) < 0.05, "双后端输出差异过大"
    print(f"[5] Separate(modelscope) OK: max1={np.abs(wm1).max():.3f} "
          f"与onnx最大差={np.abs(wm1 - w1).max():.4f}")

    # [6] 非 8k 输入自动重采样
    import torch
    fake_sr = 44100
    fake_len = int(len(w1) * fake_sr / 8000)
    fake = np.sin(2 * np.pi * 220 * np.arange(fake_len) / fake_sr).astype(np.float32)
    fake_audio = {"waveform": torch.from_numpy(fake)[None, None, :],
                  "sample_rate": fake_sr}
    r1, _ = sep.separate(fake_audio, "auto", "cpu")
    assert r1["sample_rate"] == 8000
    print(f"[6] Resample OK: {fake_sr}Hz -> 8000Hz, out_len={r1['waveform'].shape[-1]}")

    # [7] 门控后处理: hard 抑制静音段 / soft 压低
    sig = np.concatenate([np.full(8000, 0.3, np.float32), np.zeros(8000, np.float32)])
    g_hard = nodes._apply_gate(sig, "hard", 0.05)
    g_soft = nodes._apply_gate(sig, "soft", 0.05)
    assert np.abs(g_hard[10000:]).max() < 1e-3, "hard gate 未静音尾部"
    assert np.abs(g_hard[:3000]).max() > 0.2, "hard gate 误伤语音段"
    assert np.abs(g_soft[10000:]).max() < 0.05, "soft gate 未压低尾部"
    print(f"[7] Gate OK (hard 尾段峰值={np.abs(g_hard[10000:]).max():.4f}, "
          f"soft 尾段峰值={np.abs(g_soft[10000:]).max():.4f})")

    # [8] output_gain + match_input_sr
    g1, g2 = sep.separate(fake_audio, "auto", "cpu", output_order="as_is",
                          gate_mode="off", gate_threshold=0.02,
                          output_gain=2.0, match_input_sr=True)
    assert g1["sample_rate"] == fake_sr, g1["sample_rate"]
    peak_out = float(np.abs(g1["waveform"][0, 0].numpy()).max())
    assert peak_out > 0.9, f"gain 2.0 后峰值应接近 1.0，实际 {peak_out}"
    print(f"[8] Gain+MatchSR OK: 输出sr={g1['sample_rate']}, 峰值={peak_out:.3f}")

    # [9] 互斥门控 + 输出排序（固定 speaker_1）：端到端跑通且两路都是人声
    m1, m2 = sep.separate(audio, "auto", "cpu", output_order="quality",
                          gate_mode="mutual", mutual_threshold=0.25)
    wm1, wm2 = m1["waveform"][0, 0].numpy(), m2["waveform"][0, 0].numpy()
    assert m1["sample_rate"] == 8000 and m2["sample_rate"] == 8000
    assert float(np.abs(wm1).max()) > 0.01 and float(np.abs(wm2).max()) > 0.01
    # 排序后 speaker_1 的质量分应不低于 speaker_2（quality 模式硬保证）
    a1 = nodes._analyze_speech(wm1)
    a2 = nodes._analyze_speech(wm2)
    assert a1["quality"] >= a2["quality"] - 1e-6, \
        f"quality 模式下 speaker_1 质量应最高: {a1['quality']:.3f} < {a2['quality']:.3f}"
    print(f"[9] MutualGate+Ordering OK: spk1质量={a1['quality']:.3f} "
          f"spk2质量={a2['quality']:.3f} F0={a1['median_f0']:.0f}/{a2['median_f0']:.0f}Hz")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test_auto_does_not_import_modelscope()
    main()
