# HeyGem-Linux-Python-Hack 项目指南

## 项目概述

HeyGem-Linux-Python-Hack 是一个基于 Python 的数字人视频生成项目，从 [HeyGem.ai](https://github.com/GuijiAI/HeyGem.ai) 中提取并优化，能够在 Linux 系统上直接运行，摆脱了对 Docker 和 Windows 系统的依赖。

### 核心功能
- **Face-to-Face (F2F)**: 基于音频和视频生成数字人视频，让视频中的人脸根据音频内容进行口型同步
- **Text-to-Face (T2F)**: 结合 TTS（文字转语音）实现从文本直接生成数字人视频

### 技术栈
- **编程语言**: Python 3.8
- **深度学习框架**: PyTorch 1.11.0+cu113
- **推理引擎**: ONNX Runtime GPU 1.9.0
- **计算机视觉**: OpenCV
- **音频处理**: librosa, soundfile
- **语音识别**: WeNet
- **Web 界面**: Gradio
- **视频处理**: FFmpeg

### 项目架构

```
HeyGem-Linux-Python-Hack/
├── run.py                 # 命令行主程序（F2F）
├── app.py                 # Gradio Web 界面
├── download.sh            # 下载模型权重脚本
├── inference_from_text.sh # T2F 完整流程脚本
├── requirements.txt       # Python 依赖
├── config/
│   └── config.ini        # 全局配置文件
├── service/               # 数字人服务模块（.so 编译文件）
├── landmark2face_wy/      # 核心模型实现
│   ├── models/           # DINet 等模型定义
│   ├── data/             # 数据集处理
│   └── checkpoints/      # 模型权重
├── face_detect_utils/    # 人脸检测和对齐
├── face_lib/             # 人脸处理库
│   ├── face_detect_and_align/
│   ├── face_parsing/     # 人脸分割
│   └── face_restore/     # 人脸修复（GFPGAN）
├── face_attr_detect/     # 人脸属性检测
├── wenet/                # 语音识别模块
├── xseg/                 # 人脸分割
├── model_lib/            # 模型基础类
├── y_utils/              # 工具函数（配置、日志等）
├── h_utils/              # HTTP 服务工具
└── example/              # 示例文件（音频、视频）
```

## 环境要求

### 系统要求
- **操作系统**: Linux（本项目仅支持 Linux）
- **Python 版本**: 3.8（必须）
- **CUDA**: 11.3 或更高版本
- **GPU**: NVIDIA GPU（支持 CUDA）

### 关键依赖版本

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.8 | 必须使用 3.8 版本 |
| PyTorch | 1.11.0+cu113 | CUDA 11.3 版本 |
| onnxruntime-gpu | 1.9.0 | 可能需要根据 CUDA 版本调整 |
| cudatoolkit | 11.8.0 | 验证可行的 CUDA 版本 |

### 环境配置建议

如果遇到 `onnxruntime-gpu` 版本问题，建议使用以下验证可行组合：
- cudatoolkit: 11.8.0
- onnxruntime-gpu: 1.16.0

## 安装和设置

### 1. 克隆项目

```bash
git clone https://github.com/Holasyb918/HeyGem-Linux-Python-Hack
cd HeyGem-Linux-Python-Hack
```

### 2. 下载模型权重

```bash
bash download.sh
```

此脚本会下载以下模型文件：
- `face_attr_detect/face_attr_epoch_12_220318.onnx` - 人脸属性检测
- `face_detect_utils/resources/` - 人脸检测模型
- `landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth` - 核心数字人模型
- `pretrain_models/face_lib/face_parsing/79999_iter.onnx` - 人脸分割
- `pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx` - 人脸修复
- `xseg/xseg_211104_4790000.onnx` - XSeg 分割
- `wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt` - WeNet 模型

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 直接安装整个 requirements.txt 不一定成功，建议根据报错信息逐个安装依赖。

### 4. 验证环境

```bash
python check_env/check_onnx_cuda.py
```

确保输出包含 "successfully"。

### 5. 配置环境变量

如果遇到 `library.so` 找不到的错误：

```bash
# 查找库文件
sudo find /usr -name "libcublasLt.so.11"

# 如果存在，添加到环境变量
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 永久生效（添加到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 使用方法

### 方式一：命令行（Face-to-Face）

使用内置示例：

```bash
python run.py
```

使用自定义音频和视频：

```bash
python run.py --audio_path example/audio.wav --video_path example/video.mp4
```

**参数说明**:
- `--audio_path`: 音频文件路径（相对路径）
- `--video_path`: 视频文件路径（相对路径）

### 方式二：Gradio Web 界面

```bash
python app.py
```

等待模型初始化完成后，在浏览器中访问界面，上传音频和视频文件即可生成数字人视频。

### 方式三：Text-to-Face 完整流程

需要额外克隆 TTS 项目：

```bash
# 1. 克隆 TTS 项目
git clone https://github.com/Holasyb918/tts-fish-speech
cd tts-fish-speech

# 2. 下载 TTS 模型
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5/

# 3. 返回主项目
cd ../HeyGem-Linux-Python-Hack

# 4. 准备文件
# - 将参考音色放在 example/audio.wav
# - 将要生成的文本放在 example/text.txt
# - 将参考视频放在 example/video.mp4

# 5. 运行 T2F 流程
bash inference_from_text.sh example/audio.wav example/text.txt example/video.mp4
```

## 配置文件

主配置文件位于 `config/config.ini`：

```ini
[log]
log_dir = ./log
log_file = dh.log

[http_server]
server_ip = 0.0.0.0
server_port = 8383

[temp]
temp_dir = ./
clean_switch = 1

[result]
result_dir = ./result
clean_switch = 0

[digital]
batch_size = 4

[register]
url = http://172.16.160.51:12120
report_interval = 10
enable=0
```

## 核心模块说明

### 1. service/trans_dh_service
数字人转换服务的核心模块，包含 `TransDhTask` 类，负责协调整个视频生成流程。

### 2. landmark2face_wy
核心模型实现，包含：
- `DINet`: 主要的数字人生成网络
- `pirender_3dmm_mouth_hd_model`: 3DMM 嘴部高清模型
- 数据集处理和训练工具

### 3. face_detect_utils
人脸检测和对齐工具，使用 SCRFD 模型进行人脸检测。

### 4. face_lib
人脸处理库，包含：
- `face_detect_and_align`: 人脸检测和对齐
- `face_parsing`: 人脸语义分割
- `face_restore`: 基于 GFPGAN 的人脸修复

### 5. wenet
语音识别模块，用于音频特征提取。

### 6. y_utils
工具函数模块，包含：
- `config`: 全局配置管理
- `logger`: 日志记录
- `time_utils`: 时间处理工具
- `md5`: MD5 哈希计算

## 常见问题

### 1. 多个人脸报错

解决方案：下载更强大的人脸检测模型

```bash
wget https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_10g_kps.onnx
mv face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx.bak
mv scrfd_10g_kps.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx
```

### 2. ImportError: cannot import name check_argument_types

缺少依赖包：

```bash
pip install typeguard
```

### 3. onnxruntime-gpu 初始化错误

通常是 CUDA 版本不匹配。尝试：
1. 卸载 onnxruntime-gpu 和 onnxruntime
2. 使用 conda 安装对应版本的 cudatoolkit
3. 重新安装 onnxruntime-gpu

验证可行版本：
- cudatoolkit 11.8.0 + onnxruntime-gpu 1.16.0

### 4. library.so 找不到

见上文"配置环境变量"部分。

## 开发约定

### 代码风格
- Python 版本严格限制为 3.8
- 使用类型提示（typing）
- 遵循 PEP 8 编码规范

### 日志记录
使用 `y_utils.logger.logger` 进行日志记录：

```python
from y_utils.logger import logger

logger.info("信息日志")
logger.error("错误日志")
```

### 配置管理
使用 `y_utils.config.GlobalConfig` 获取全局配置：

```python
from y_utils.config import GlobalConfig

config = GlobalConfig.instance()
result_dir = config.result_dir
```

### 错误处理
使用 `h_utils.custom.CustomError` 进行自定义错误处理：

```python
from h_utils.custom import CustomError

raise CustomError("错误描述")
```

## 模型文件位置

| 模型类型 | 位置 |
|---------|------|
| 人脸属性检测 | `face_attr_detect/face_attr_epoch_12_220318.onnx` |
| 人脸检测 | `face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx` |
| 关键点检测 | `face_detect_utils/resources/pfpld_robust_sim_bs1_8003.onnx` |
| 核心数字人模型 | `landmark2face_wy/checkpoints/anylang/dinet_v1_20240131.pth` |
| 人脸分割 | `pretrain_models/face_lib/face_parsing/79999_iter.onnx` |
| 人脸修复 | `pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx` |
| XSeg 分割 | `xseg/xseg_211104_4790000.onnx` |
| WeNet 模型 | `wenet/examples/aishell/aidata/exp/conformer/wenetmodel.pt` |

## 输出说明

生成的视频文件保存在 `result/` 目录下，文件名格式为：
- 临时文件: `{work_id}-t.mp4`
- 结果文件: `{work_id}-r.mp4`

## 依赖说明

本项目使用大量编译的 `.so` 文件，这些文件由硅基（GuijiAI）编译，与开发者无关。所有模型也由硅基提供。

## 许可证

参考 HeyGem.ai 的协议。

## 相关链接

- [HeyGem.ai 原项目](https://github.com/GuijiAI/HeyGem.ai)
- [RTX 50 版本](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack-RTX-50)
- [TTS-Fish-Speech](https://github.com/Holasyb918/tts-fish-speech)
- [AutoDL 环境参考](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/issues/43)

## 注意事项

1. **Python 版本**: 必须使用 Python 3.8，不支持其他版本
2. **操作系统**: 仅支持 Linux，不支持 Windows
3. **离线运行**: 项目完全支持离线运行
4. **GPU 要求**: 需要 NVIDIA GPU 并安装 CUDA
5. **文件路径**: 命令行参数仅支持相对路径
6. **模型下载**: 首次使用必须运行 `download.sh` 下载模型
7. **环境搭建**: 如果环境搭建困难，建议参考 AutoDL 环境