# ComfyUI-FlatSepReformer

[FLASepformer](https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100)
（FLA-SepReformer-B）双说话人语音分离模型的 **ComfyUI 自定义节点**。

模型在 Libri2Mix-100 上训练，用于 **8 kHz 单声道双说话人语音分离**：
输入一段两人混合语音，输出两路独立的说话人语音。

> 模型许可证：CC BY-NC 4.0（非商业用途）。模型权重不随本仓库分发，
> 请运行 `python install.py` 自动下载到 `<ComfyUI>/models/FlatSepReformer/`。

---

## ✨ 特性

- 🎛 **3 个即插即用节点**：分离节点已合并"模型加载 + 语音分离"，开箱即用
- 🚀 **双推理后端（auto 自动选择）**：
  - `onnx`：使用 `onnx_model.onnx`，基于 ONNX Runtime，**无需安装 modelscope**（默认优先）
  - `modelscope`：使用 `pytorch_model.pt`（需 master 源码版）
- 📦 **模型自动定位**：默认读取 `<ComfyUI>/models/FlatSepReformer/`
  （基于插件位置的相对路径推导，无需填写任何路径）
- 🔄 **自动重采样**：任意采样率的输入音频自动重采样到 8000 Hz
- 💾 **模型缓存**：同一配置只加载一次，重复执行不重复占显存
- 🎚 **双后端输出一致**：ONNX 与 modelscope 后端均按峰值归一化到 0.5（与官方 pipeline 后处理一致），相关系数 > 0.9999

## 📦 安装

### 1. 克隆插件到 ComfyUI

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/<你的用户名>/ComfyUI-FlatSepReformer.git
```

### 2. 安装 Python 依赖

```bash
cd ComfyUI-FlatSepReformer
# 建议使用 ComfyUI 的 python_embeded 环境（Windows: python_embeded\python.exe -m pip ...）
pip install -r requirements.txt
```

> ⚠️ ModelScope 后端要求 **最新 master 源码**（模型卡要求）：
>
> ```bash
> pip install -U "modelscope @ git+https://github.com/modelscope/modelscope.git@master"
> ```
>
> 如果只想用 `onnx` 后端，可跳过 modelscope。

### 3. 下载模型到指定目录

```bash
python install.py
# 或手动指定 ComfyUI 根目录
COMFYUI_PATH=D:/ComfyUI python install.py
```

模型会下载到 **`<ComfyUI>/models/FlatSepReformer/`**（约 60 MB）：
`configuration.json`、`pytorch_model.pt`、`onnx_model.onnx`、`README.md` 等。

> 也可手动从 [ModelScope](https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100)
> 下载模型文件放入该目录；或设置环境变量 `FLATSEPREFORMER_MODEL_DIR` 指向自定义目录。

### 4. 重启 ComfyUI

重启后在节点菜单 **`audio/separation`** 分类下即可看到全部节点。

## 🧩 节点说明

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| **FlatSepReformer (Separate 2 Speakers)** | `audio`、`backend`、`device` | `speaker_1`、`speaker_2` | **合并节点**：自动加载 `<ComfyUI>/models/FlatSepReformer` 模型并分离双说话人。`backend` 默认 `auto`（优先 onnx，无需 modelscope） |
| **Load Audio (FlatSepReformer)** | `audio_path` | `audio` | 加载 wav/flac/ogg/mp3 等格式音频文件 |
| **Save Audio (FlatSepReformer)** | `audio`、`filename`、`output_dir` | `filepath` | 保存为 wav，返回保存路径（默认 `<ComfyUI>/output/`） |

> v2 变更：原 `FlatSepReformerLoader` + `FlatSepReformerSeparate` 两个节点已合并为
> `FlatSepReformerSeparate` 单节点（输入 `audio`，自动加载模型）。请删除工作流中旧的
> Loader 节点，直接用新的分离节点。

`AUDIO` 类型遵循社区通用约定：`{"waveform": torch.Tensor [B, C, T], "sample_rate": int}`，
可与其它音频节点（如 ComfyUI-Audio 生态）互相连接。

## 🚀 快速上手

1. **Load Audio**：填入混合语音 wav 路径（两人同时说话）
2. **Separate**：连接 audio，`backend` 留 `auto`、`device` 留 `auto` 即可（自动加载模型）
3. **Save Audio ×2**：分别接 `speaker_1`、`speaker_2`，设置文件名

`examples/workflow.json` 提供可直接导入的 API 格式示例工作流。

## ✅ 测试

```bash
cd ComfyUI-FlatSepReformer
python tests/smoke_test.py
```

冒烟测试覆盖：音频加载 → 模型加载（onnx + modelscope 双后端）→ 双说话人分离 → 音频保存 → 非 8k 输入自动重采样。
测试通过标志：`ALL TESTS PASSED`。

## 🖥 后端选择建议

| 场景 | 推荐 backend |
|---|---|
| 默认（推荐） | `auto`：模型目录有 `onnx_model.onnx` 时自动用 onnx，无需 modelscope |
| 已安装 master 版 modelscope | `modelscope`（与 ModelScope 生态一致） |
| 只想用 PyTorch 权重 | `modelscope`（需先升级 modelscope，见常见问题） |

> ⚠️ 默认 `auto` 优先 ONNX，**完全不需要 modelscope**，可避免
> "`... is not in the pipelines registry group speech-separation`" 的版本问题。

## ⚙️ 模型目录与相对路径

节点通过**插件自身位置的相对路径**自动定位模型（不依赖工作目录、无需填绝对路径）：

```
<ComfyUI>/
├── models/
│   └── FlatSepReformer/          ← 模型目录（configuration.json / pytorch_model.pt / onnx_model.onnx）
└── custom_nodes/
    └── ComfyUI-FlatSepReformer/  ← 插件位置（相对推导 ../../models/FlatSepReformer）
```

如需自定义位置：环境变量 `FLATSEPREFORMER_MODEL_DIR`（优先级高于自动探测）。

```bash
# Windows PowerShell
$env:FLATSEPREFORMER_MODEL_DIR = "D:/models/FlatSepReformer"
# 重启 ComfyUI
```

## 🛠 常见问题

**Q: 报错 `... is not in the pipelines registry group speech-separation`**
modelscope 是 PyPI 版，其 pipeline 注册表还没有这个新模型。两种解法：
1. **推荐**：节点 `backend` 选 `auto` 或 `onnx`（模型自带 `onnx_model.onnx`，无需 modelscope）
2. 升级 modelscope 到 master 源码版：`pip install -U "modelscope @ git+https://github.com/modelscope/modelscope.git@master"`
   （若网络访问不了 GitHub，请用解法 1）

**Q: 提示「未找到模型目录」**
模型未下载或不在默认位置，运行 `python install.py` 或检查 `<ComfyUI>/models/FlatSepReformer` 是否包含 `configuration.json`。

**Q: modelscope 后端报错 / 模型不识别**
确认安装了 master 源码版 modelscope（见上文）。也可直接切换到 `onnx` 后端。

**Q: 输入采样率不是 8000 Hz 可以吗？**
可以，节点会自动重采样到 8000 Hz（模型要求）。

**Q: 输出音量/长度有变化？**
分离输出为 8 kHz 单声道 wav；模型为干净双说话人分离，不含降噪功能。

## 📄 许可证

- 插件代码：MIT License
- 模型权重：CC BY-NC 4.0（见 [ModelScope 模型卡](https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100)）

## 🙏 致谢

- 模型：[ModelScope iic / speech_flatsepreformer_separation_temporal_8k_base_libri2mix100](https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
