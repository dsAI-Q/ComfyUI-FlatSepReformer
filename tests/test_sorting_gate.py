# -*- coding: utf-8 -*-
"""
新增能力单元测试（无需模型，纯 numpy）：
  - _apply_mutual_gate : 交替对话互斥门控（抑制另一路泄漏、不误伤重叠段）
  - _analyze_speech    : 质量分（区分人声/噪声）+ 性别（基频）检测
  - _decide_order      : 输出排序（固定 speaker_1）

用法:
    cd ComfyUI-FlatSepReformer
    python tests/test_sorting_gate.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nodes  # noqa: E402

FS = nodes.TARGET_SAMPLE_RATE


def _tone(freq: float, dur_s: float, amp: float = 0.3) -> np.ndarray:
    n = int(FS * dur_s)
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / FS)).astype(np.float32)


def test_mutual_gate_alternating() -> None:
    """交替对话：某路占优时应压低另一路的泄漏，同时保留自身语音段。"""
    rng = np.random.default_rng(0)
    leak = 0.02  # 泄漏幅度（RMS 约 0.02，约为语音 RMS 的 1/10）
    # w1: 前 1s 说话，后 1s 只有泄漏；w2: 前 1s 只有泄漏，后 1s 说话
    w1 = np.zeros(2 * FS, np.float32)
    w2 = np.zeros(2 * FS, np.float32)
    w1[:FS] = _tone(180, 1.0)
    w2[FS:] = _tone(220, 1.0)
    w1[FS:] = (leak * rng.standard_normal(FS)).astype(np.float32)
    w2[:FS] = (leak * rng.standard_normal(FS)).astype(np.float32)

    g1, g2 = nodes._apply_mutual_gate(w1, w2, 0.25)

    # 自身语音段应保留
    assert np.abs(g1[:FS]).max() > 0.2, f"w1 语音段被误伤: {np.abs(g1[:FS]).max():.3f}"
    assert np.abs(g2[FS:]).max() > 0.2, f"w2 语音段被误伤: {np.abs(g2[FS:]).max():.3f}"
    # 泄漏段稳态（跳过 ~150ms 转换区）应被明显抑制（原峰值 ~0.08 → < 0.01）
    assert np.abs(g1[FS + FS // 10:]).max() < 0.01, \
        f"w1 尾段泄漏未抑制: {np.abs(g1[FS + FS // 10:]).max():.4f}"
    assert np.abs(g2[:FS - FS // 10]).max() < 0.01, \
        f"w2 头段泄漏未抑制: {np.abs(g2[:FS - FS // 10]).max():.4f}"
    # 泄漏段整体 RMS 显著下降（原 ~0.02 → < 0.006）
    assert float(np.sqrt(np.mean(g1[FS:] ** 2))) < 0.006, "w1 泄漏整体 RMS 偏高"
    assert float(np.sqrt(np.mean(g2[:FS] ** 2))) < 0.006, "w2 泄漏整体 RMS 偏高"
    print(f"[mutual] 交替对话抑制 OK（泄漏峰值 ~0.08 → 稳态 "
          f"{np.abs(g1[FS + FS // 10:]).max():.4f}，语音段保留）")


def test_mutual_gate_overlap_kept() -> None:
    """重叠说话：两路能量相当，不应互相抑制。"""
    w1 = _tone(180, 1.0)
    w2 = _tone(220, 1.0)
    g1, g2 = nodes._apply_mutual_gate(w1, w2, 0.25)
    assert np.abs(g1).max() > 0.2, "重叠段 spk1 被误伤"
    assert np.abs(g2).max() > 0.2, "重叠段 spk2 被误伤"
    # 与原始信号相关性应很高
    c1 = float(np.corrcoef(w1[:FS], g1[:FS])[0, 1])
    c2 = float(np.corrcoef(w2[:FS], g2[:FS])[0, 1])
    assert c1 > 0.9 and c2 > 0.9, f"重叠段被过度改动: corr={c1:.3f}/{c2:.3f}"
    print(f"[mutual] 重叠段保留 OK（corr={c1:.3f}/{c2:.3f}）")


def test_quality_noise_vs_speech() -> None:
    """质量分：干净人声明显高于噪声。"""
    tone = _tone(200, 1.5)
    rng = np.random.default_rng(1)
    noise = (0.3 * rng.standard_normal(len(tone))).astype(np.float32)
    qt = nodes._analyze_speech(tone)["quality"]
    qn = nodes._analyze_speech(noise)["quality"]
    assert qt > 0.7, f"人声质量分偏低: {qt:.3f}"
    assert qn < 0.5, f"噪声质量分偏高: {qn:.3f}"
    assert qt > qn + 0.3
    print(f"[quality] 人声 {qt:.3f} vs 噪声 {qn:.3f} OK")


def test_gender_detection() -> None:
    """性别（基频）检测：120Hz 判男声，250Hz 判女声。"""
    male = nodes._analyze_speech(_tone(120, 1.0))
    female = nodes._analyze_speech(_tone(250, 1.0))
    assert male["is_female"] is False, f"120Hz 应判男声: {male}"
    assert female["is_female"] is True, f"250Hz 应判女声: {female}"
    assert 100 < male["median_f0"] < 150, f"男声 F0 异常: {male['median_f0']}"
    assert 200 < female["median_f0"] < 300, f"女声 F0 异常: {female['median_f0']}"
    print(f"[gender] 男 F0={male['median_f0']:.0f}Hz 女 F0={female['median_f0']:.0f}Hz OK")


def test_decide_order() -> None:
    """排序：质量优先；质量接近时女声优先到 speaker_1。"""
    clean = {"quality": 0.9, "is_female": None}
    bad = {"quality": 0.2, "is_female": None}
    # auto: 质量差距明显 → speaker_1 = 好的一路
    assert nodes._decide_order(bad, clean, "auto") is True, "auto 应交换使好音在 spk1"
    assert nodes._decide_order(clean, bad, "auto") is False
    # auto: 质量接近 → 女声在 spk1
    m = {"quality": 0.85, "is_female": True}
    f = {"quality": 0.84, "is_female": True}
    assert nodes._decide_order(f, m, "auto") is False, "质量接近女声已在 spk1 不交换"
    # 男声在前、女声在后 → 交换
    male_m = {"quality": 0.85, "is_female": False}
    female_m = {"quality": 0.84, "is_female": True}
    assert nodes._decide_order(male_m, female_m, "auto") is True, "女声应换到 spk1"
    # quality 模式只按质量
    assert nodes._decide_order(male_m, female_m, "quality") is False
    # as_is 永不交换
    assert nodes._decide_order(bad, clean, "as_is") is False
    print("[order] auto/quality/as_is 排序判定 OK")


def test_short_input_safe() -> None:
    """超短输入不应崩溃。"""
    r = nodes._analyze_speech(np.zeros(100, np.float32))
    assert r["quality"] == 0.0 and r["is_female"] is None
    g1, g2 = nodes._apply_mutual_gate(np.zeros(50, np.float32),
                                      np.zeros(50, np.float32), 0.25)
    assert len(g1) == 50 and len(g2) == 50
    print("[edge] 短输入安全 OK")


if __name__ == "__main__":
    test_mutual_gate_alternating()
    test_mutual_gate_overlap_kept()
    test_quality_noise_vs_speech()
    test_gender_detection()
    test_decide_order()
    test_short_input_safe()
    print("\nALL SORTING/GATE TESTS PASSED")
