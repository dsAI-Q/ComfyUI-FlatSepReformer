# -*- coding: utf-8 -*-
"""
一键下载 FLASepformer 模型到 <ComfyUI>/models/FlatSepReformer/

用法:
    python install.py
    # 或指定 ComfyUI 根目录:
    COMFYUI_PATH=D:/ComfyUI python install.py
"""

import os
import sys

MODEL_ID = "iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100"
MODEL_DIR_NAME = "FlatSepReformer"


def find_comfyui_root() -> str:
    env = os.environ.get("COMFYUI_PATH", "").strip()
    if env and os.path.isdir(env):
        return env

    # 插件位于 <root>/custom_nodes/ComfyUI-FlatSepReformer/install.py
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    grandparent = os.path.dirname(parent)
    if os.path.basename(parent) == "custom_nodes" and \
            os.path.isfile(os.path.join(grandparent, "main.py")):
        return grandparent

    # 向上查找包含 main.py + models 的目录
    cur = here
    while True:
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
        if os.path.isfile(os.path.join(cur, "main.py")) and \
                os.path.isdir(os.path.join(cur, "models")):
            return cur
    return ""


def main() -> None:
    root = find_comfyui_root()
    if root:
        target = os.path.join(root, "models", MODEL_DIR_NAME)
    else:
        target = os.path.join(os.getcwd(), "models", MODEL_DIR_NAME)
    os.makedirs(target, exist_ok=True)

    print(f"模型: {MODEL_ID}")
    print(f"下载到: {os.path.abspath(target)}")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("\n[错误] 缺少 modelscope，请先执行: pip install -r requirements.txt")
        sys.exit(1)

    snapshot_download(MODEL_ID, local_dir=target)
    print(f"\n完成! 模型已下载到: {os.path.abspath(target)}")
    print("在 ComfyUI 节点中留空 model_dir 即可自动加载。")
    print("如需自定义位置，可设置环境变量 FLATSEPREFORMER_MODEL_DIR 后重启 ComfyUI。")


if __name__ == "__main__":
    main()
