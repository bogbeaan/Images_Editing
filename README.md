# 在p了小组展示
国科大人工智能安全与对抗大作业
# XceptionWithViT 模型文档：
## 项目简介
本项目实现了一个结合Xception和ViT的深度学习模型，用于图像分类任务。
## 主要功能
本模型将Xception网络与Vision Transformer结合，用于深度伪造检测任务。主要特点：
•	基于Xception的特征提取能力
•	融入4层ViT模块增强全局特征理解
•	适用于二分类任务（真实/伪造）
## 代码结构
•	SeparableConv2d：深度可分离卷积层
•	Block：Xception的基本构建块
•	Xception：Xception模型主体
•	MultiHeadAttention：多头注意力机制
•	LayerNormalization：层归一化
•	ViTBlock：ViT的基本构建块
•	XceptionWithViT：结合Xception和ViT的模型
•	vitxception：模型构造函数
## 使用方法
1.	环境准备
Python 3.8+
PyTorch 1.12.1+
torchvision 0.13.0+
CUDA 11.6（推荐）
确保已安装Python和PyTorch库。可以使用以下命令安装必要的库：
pip install torch torchvision
2.	代码导入
在Python脚本中导入模型：
from your_module import vitxception
3.	模型初始化
model = vitxception(pretrained=False, num_classes=1)
4.	前向传播
input_tensor = torch.rand(1, 3, 299, 299)
output = model(input_tensor)
print(output)
## 注意事项
•	如果需要使用预训练权重，请确保提供正确的权重文件路径，并在模型初始化时设置pretrained=True
•	根据您的具体任务调整num_classes参数
•	如果您需要从Dropbox链接加载预训练权重但遇到问题，请检查链接的有效性并确保网络连接正常
## 示例
以下是一个完整的示例：
import torch
### 初始化模型
model = vitxception(pretrained=False, num_classes=1)
outputs = model(inputs)


# EfficientNetV2 模型文档
## 模型概述
基于EfficientNetV2的深度伪造检测模型，主要特点：
•	使用timm库预训练模型
•	增强分类头设计
•	支持在线/离线权重加载
•	更高准确率和推理速度
## 文件结构
├── efficient_model.py # 模型定义核心代码
├── README.md          # 本文档
└── requirements.txt   # 依赖库
## 依赖环境
•	Python 3.8+
•	PyTorch 1.12.1+
•	timm 0.6.0+
•	CUDA 11.6（推荐）
确保已安装Python和PyTorch库。可以使用以下命令安装必要的库：
pip install torch torchvision timm
## 模型结构
1.	Backbone：EfficientNetV2-S
o	输入尺寸：384x384x3（自适应）
o	预训练权重：ImageNet-21k+1k
2.	分类头：
nn.Sequential(
    nn.Linear(1792, 1280),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(1280, num_classes),
    nn.Sigmoid()
)
## 使用方法
from efficient_model import vitxception

### 在线加载预训练
model = efficientnet(pretrained=True)

### 加载本地权重
model = efficientnet (pretrained="/path/to/weights.pth")

### 随机初始化
model = efficientnet (pretrained=False)
outputs = model(inputs)
## 权重加载机制
•	在线加载 ：当pretrained=True时，尝试从在线资源加载预训练权重。
•	本地加载 ：当pretrained为字符串路径时，尝试从指定路径加载本地权重文件。
•	随机初始化 ：当pretrained=False或权重加载失败时，使用随机初始化创建模型。
通过这种灵活的权重加载机制，用户可以根据实际需求选择合适的模型初始化方式。

