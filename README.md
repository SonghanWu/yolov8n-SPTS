# YOLOv8n-SPTS: Traffic Scene Small Target Detection Method for Autonomous Driving

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](./Traffic%20Scene%20Small%20Target%20Detection%20Method%20Based%20on%20YOLOv8n-SPTS%20Model%20for%20Autonomous%20Driving.pdf)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)

[English](#english) | [中文](#中文)

</div>

---

## English

### 📋 Overview

**YOLOv8n-SPTS** is an enhanced version of YOLOv8 specifically designed for **small target detection in traffic scenes for autonomous driving**. SPTS stands for **Spatial Pyramid Transformer with Self-attention**, which integrates multiple attention mechanisms to capture multi-scale features more effectively, particularly for detecting small objects in complex traffic environments.

This project is based on the paper: **"Traffic Scene Small Target Detection Method Based on YOLOv8n-SPTS Model for Autonomous Driving"**.

### ✨ Key Features

- 🎯 **Multiple Attention Mechanisms Integration**
  - **CFF Attention** (Channel Feature Fusion): Fuses features from different channels using max and average pooling
  - **ECA Attention** (Efficient Channel Attention): Lightweight channel attention with 1D convolution
  - **SE Attention** (Squeeze-and-Excitation): Classic channel attention mechanism
  - **Shuffle Attention**: Combines channel and spatial attention with channel shuffling

- ⚡ **Enhanced Detection Performance**
  - Improved feature extraction capabilities for small targets
  - Better multi-scale object detection in traffic scenes
  - Enhanced feature representation through attention fusion
  - Optimized for autonomous driving scenarios

- 🚗 **Traffic Scene Specialization**
  - Designed for small object detection (pedestrians, traffic signs, distant vehicles)
  - Robust performance in complex traffic environments
  - Real-time detection capability for autonomous driving
  - High accuracy on occluded and partially visible objects

- 🔧 **Complete Framework**
  - Full training, validation, and prediction pipeline
  - Compatible with YOLOv8 ecosystem
  - Easy to use Python API and CLI
  - Pre-trained models available via Git LFS

### 🏗️ Architecture

The project implements four attention mechanisms that work together in the YOLOv8 backbone:

```
YOLOv8 Backbone
    ├── CFF Attention Module
    ├── ECA Attention Module
    ├── SE Attention Module
    └── Shuffle Attention Module
```

Each attention module enhances the feature representation at different scales, leading to improved detection accuracy.

### 📊 Training Results

Based on the training logs in `runs/detect/train2/`:

- **mAP50**: 0.010 → 0.501 (50x improvement)
- **mAP50-95**: 0.004 → 0.334 (83x improvement)
- **Precision**: 0.008 → 0.674
- **Recall**: 0.214 → 0.533

### 🚀 Quick Start

#### Installation

1. **Clone the repository (with Git LFS for model files)**
```bash
# Install Git LFS first (if not already installed)
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs
# Windows: Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Clone the repository with model files
git clone git@github.com:SonghanWu/yolov8n-SPTS.git
cd yolov8n-SPTS

# Pull LFS files (model weights and paper PDF)
git lfs pull
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
# or
pip install ultralytics
```

#### Download Pre-trained Models

The pre-trained model weights are stored using Git LFS in the `models/` directory:
- `YOLOv8n-SPTS.pt` - Main SPTS model
- `yolov8-CFF.pt` - CFF Attention variant
- `yolov8-ECA.pt` - ECA Attention variant
- `yolov8-SE.pt` - SE Attention variant
- `yolov8-SA.pt` - Shuffle Attention variant

If you cloned without Git LFS, download models manually from [Releases](https://github.com/SonghanWu/yolov8n-SPTS/releases).

#### Usage

##### Python API

```python
from ultralytics import YOLO

# Load the SPTS model
model = YOLO("path/to/YOLOv8n-SPTS.pt")

# Predict on an image
results = model("path/to/image.jpg")

# Display results
results[0].show()

# Save results
results[0].save("output.jpg")
```

##### Training

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.yaml")

# Train with custom dataset
model.train(
    data="your_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

##### Command Line

```bash
# Prediction
yolo predict model=YOLOv8n-SPTS.pt source=image.jpg

# Training
yolo train model=yolov8n.yaml data=dataset.yaml epochs=100
```

### 📁 Project Structure

```
yolov8n-SPTS/
├── ultralytics/
│   ├── nn/
│   │   ├── CFFAttention.py      # CFF attention module
│   │   ├── ECAAttention.py      # ECA attention module
│   │   ├── SEAttention.py       # SE attention module
│   │   └── ShuffleAttention.py  # Shuffle attention module
│   └── ...
├── models/                       # Pre-trained model directory
├── runs/detect/                  # Training results
├── Train_model.py                # Training script
└── README.md
```

### 🔬 Attention Mechanisms Details

#### 1. CFF Attention (Channel Feature Fusion)
Combines features from different channels using both max pooling and average pooling, then applies 2D convolution for feature fusion.

#### 2. ECA Attention (Efficient Channel Attention)
Uses 1D convolution for efficient channel attention computation, reducing parameters while maintaining performance.

#### 3. SE Attention (Squeeze-and-Excitation)
Classic attention mechanism that uses global average pooling followed by fully connected layers to learn channel-wise attention weights.

#### 4. Shuffle Attention
Splits features into groups, applies channel and spatial attention separately, then shuffles channels for better information flow.

### 📈 Performance Comparison

| Model | mAP50 | mAP50-95 | Params | FLOPs |
|-------|-------|----------|--------|-------|
| YOLOv8n | 37.3 | - | 3.2M | 8.7B |
| YOLOv8n-SPTS | **50.1** | **33.4** | - | - |

### 📝 Citation

If you use this project in your research, please consider citing:

```bibtex
@misc{yolov8n-spts,
  title={YOLOv8n-SPTS: YOLOv8 with Multiple Attention Mechanisms},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/SonghanWu/yolov8n-SPTS}}
}
```

### 📄 License

This project is based on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and follows the AGPL-3.0 license.

### 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base framework
- All attention mechanism papers and implementations

---

## 中文

### 📋 项目简介

**YOLOv8n-SPTS** 是专门为**自动驾驶交通场景中的小目标检测**设计的 YOLOv8 增强版本。SPTS 代表**空间金字塔变换器与自注意力机制**，集成了多种注意力机制来更有效地捕获多尺度特征，特别是针对复杂交通环境中的小目标检测进行了优化。

本项目基于论文：**《基于YOLOv8n-SPTS模型的自动驾驶交通场景小目标检测方法》**。

### ✨ 核心特性

- 🎯 **多种注意力机制集成**
  - **CFF 注意力**（通道特征融合）：使用最大池化和平均池化融合不同通道的特征
  - **ECA 注意力**（高效通道注意力）：使用1D卷积的轻量级通道注意力
  - **SE 注意力**（挤压激励）：经典的通道注意力机制
  - **Shuffle 注意力**：结合通道和空间注意力，通过通道混洗增强特征交互

- ⚡ **增强的检测性能**
  - 针对小目标改进的特征提取能力
  - 交通场景中更好的多尺度目标检测
  - 通过注意力融合增强特征表示
  - 针对自动驾驶场景优化

- 🚗 **交通场景专业化**
  - 专为小目标检测设计（行人、交通标志、远处车辆）
  - 在复杂交通环境中表现稳健
  - 满足自动驾驶实时检测需求
  - 对遮挡和部分可见物体具有高精度

- 🔧 **完整框架**
  - 完整的训练、验证和预测流程
  - 与 YOLOv8 生态系统兼容
  - 易用的 Python API 和命令行接口
  - 通过 Git LFS 提供预训练模型

### 🏗️ 网络架构

项目实现了四种注意力机制，它们在 YOLOv8 骨干网络中协同工作：

```
YOLOv8 骨干网络
    ├── CFF 注意力模块
    ├── ECA 注意力模块
    ├── SE 注意力模块
    └── Shuffle 注意力模块
```

每个注意力模块在不同尺度上增强特征表示，从而提高检测精度。

### 📊 训练结果

基于 `runs/detect/train2/` 中的训练日志：

- **mAP50**: 0.010 → 0.501（提升50倍）
- **mAP50-95**: 0.004 → 0.334（提升83倍）
- **精确率**: 0.008 → 0.674
- **召回率**: 0.214 → 0.533

### 🚀 快速开始

#### 安装

1. **克隆仓库（使用 Git LFS 下载模型文件）**
```bash
# 首先安装 Git LFS（如果尚未安装）
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs
# Windows: 从 https://git-lfs.github.com/ 下载

# 初始化 Git LFS
git lfs install

# 克隆仓库及模型文件
git clone git@github.com:SonghanWu/yolov8n-SPTS.git
cd yolov8n-SPTS

# 拉取 LFS 文件（模型权重和论文PDF）
git lfs pull
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows 系统: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
# 或者
pip install ultralytics
```

#### 下载预训练模型

预训练模型权重通过 Git LFS 存储在 `models/` 目录中：
- `YOLOv8n-SPTS.pt` - 主要 SPTS 模型
- `yolov8-CFF.pt` - CFF 注意力变体
- `yolov8-ECA.pt` - ECA 注意力变体
- `yolov8-SE.pt` - SE 注意力变体
- `yolov8-SA.pt` - Shuffle 注意力变体

如果克隆时未使用 Git LFS，请从 [Releases](https://github.com/SonghanWu/yolov8n-SPTS/releases) 手动下载模型。

#### 使用方法

##### Python API

```python
from ultralytics import YOLO

# 加载 SPTS 模型
model = YOLO("path/to/YOLOv8n-SPTS.pt")

# 对图片进行预测
results = model("path/to/image.jpg")

# 显示结果
results[0].show()

# 保存结果
results[0].save("output.jpg")
```

##### 训练模型

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")

# 使用自定义数据集训练
model.train(
    data="your_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

##### 命令行

```bash
# 预测
yolo predict model=YOLOv8n-SPTS.pt source=image.jpg

# 训练
yolo train model=yolov8n.yaml data=dataset.yaml epochs=100
```

### 📁 项目结构

```
yolov8n-SPTS/
├── ultralytics/
│   ├── nn/
│   │   ├── CFFAttention.py      # CFF 注意力模块
│   │   ├── ECAAttention.py      # ECA 注意力模块
│   │   ├── SEAttention.py       # SE 注意力模块
│   │   └── ShuffleAttention.py  # Shuffle 注意力模块
│   └── ...
├── models/                       # 预训练模型目录
├── runs/detect/                  # 训练结果
├── Train_model.py                # 训练脚本
└── README.md
```

### 🔬 注意力机制详解

#### 1. CFF 注意力（通道特征融合）
结合最大池化和平均池化来融合不同通道的特征，然后应用2D卷积进行特征融合。

#### 2. ECA 注意力（高效通道注意力）
使用1D卷积进行高效的通道注意力计算，在保持性能的同时减少参数量。

#### 3. SE 注意力（挤压激励）
经典的注意力机制，使用全局平均池化和全连接层来学习通道级的注意力权重。

#### 4. Shuffle 注意力
将特征分组，分别应用通道注意力和空间注意力，然后混洗通道以增强信息流动。

### 📈 性能对比

| 模型 | mAP50 | mAP50-95 | 参数量 | FLOPs |
|-------|-------|----------|--------|-------|
| YOLOv8n | 37.3 | - | 3.2M | 8.7B |
| YOLOv8n-SPTS | **50.1** | **33.4** | - | - |

### 📝 引用

如果您在研究中使用了本项目，请考虑引用：

```bibtex
@misc{yolov8n-spts,
  title={YOLOv8n-SPTS: YOLOv8 with Multiple Attention Mechanisms},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/SonghanWu/yolov8n-SPTS}}
}
```

### 📄 许可证

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，遵循 AGPL-3.0 许可证。

### 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 提供的基础框架
- 所有注意力机制的论文和实现

---

<div align="center">

**如果这个项目对您有帮助，请给个 ⭐️ Star 支持一下！**

**If this project helps you, please give it a ⭐️ Star!**

</div>
