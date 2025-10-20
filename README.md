# Kronos BTC 微调预测模型

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/lc2004)


**基于 Kronos 金融预测模型的 BTC/USDT 微调预测**

</div>

## 📖 项目简介

本项目提供基于 [Kronos](https://github.com/shiyu-coder/Kronos) 金融预测模型针对 **BTC/USDT** 交易对的微调模型和预测结果。Kronos 是第一个开源的金融K线（蜡烛图）基础模型，在45个全球交易所的数据上进行训练。

**项目特点**：
- 🎯 专注于微调结果和预测应用
- 📦 包含完整的 Kronos 框架（与官方仓库保持同步）
- 🚀 即插即用的预测脚本
- 📊 提供微调后的模型权重和预测结果
- 🤗 模型已上传至 Hugging Face，便于下载使用

**注意**：本项目不公开微调的具体细节和训练过程，仅提供可用的模型和预测脚本。

## ℹ️ 项目版本

- **当前版本**：1.0.0
- **更新时间**：2025年10月20日
- **项目定位**：微调结果发布
- **最新更新**：
  - ✅ 更新项目定位为微调结果展示
  - ✅ 简化文档，移除内部细节说明
  - ✅ 保留完整 Kronos 框架供预测使用
  - ✅ 新增 Hugging Face 模型下载链接

## ✨ 核心功能

- **实时预测**：基于微调模型进行 BTC/USDT 价格预测
- **开箱即用**：无需微调，直接使用预训练模型
- **可视化结果**：自动生成预测图表
- **数据获取工具**：包含 BTC 数据爬取脚本
- **云端模型**：从 Hugging Face 一键下载

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 支持（推荐，GPU可提升预测速度）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载模型

模型已上传至 Hugging Face，请从以下链接下载：

#### 🤗 方法一：使用 Hugging Face CLI（推荐）

```bash
# 下载微调模型
huggingface-cli download lc2004/kronos_base_model_BTCUSDT_1h_finetune --local-dir ./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/basemodel/best_model

# 下载 Tokenizer
huggingface-cli download lc2004/kronos_tokenizer_base_BTCUSDT_1h_finetune --local-dir ./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/tokenizer/best_model
```

#### 🌐 方法二：手动下载

从以下 Hugging Face 仓库手动下载：

1. **微调模型**：[lc2004/kronos_base_model_BTCUSDT_1h_finetune](https://huggingface.co/lc2004/kronos_base_model_BTCUSDT_1h_finetune)
   - 放置到：`./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/basemodel/best_model`

2. **Tokenizer**：[lc2004/kronos_tokenizer_base_BTCUSDT_1h_finetune](https://huggingface.co/lc2004/kronos_tokenizer_base_BTCUSDT_1h_finetune)
   - 放置到：`./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/tokenizer/best_model`

### 运行预测

模型下载完成后，直接运行预测脚本获取 BTC 价格预测：

```bash
python btc_prediction.py
```

预测结果将自动保存到输出目录。

## 🔧 详细使用说明

### 预测脚本参数

预测脚本会自动加载下载的模型。如需修改参数，编辑 `btc_prediction.py`：

```python
# 模型路径（确保已下载到此路径）
tokenizer_path = "./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/tokenizer/best_model"
model_path = "./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/basemodel/best_model"

# 预测参数
lookback_window = 512        # 历史数据窗口
pred_len = 48                # 预测长度
sample_count = 5             # 采样次数
```

### 获取数据

如需更新 BTC 数据：

```bash
cd get_btc_data
python get_Data_of_all.py          # 获取历史数据
# 或
python get_Data_of_realtime.py     # 实时监控数据
```

## 📁 项目结构

```
Kronos-Btc-finetune/
├── btc_prediction.py              # 预测脚本
├── requirements.txt               # 依赖
├── README.md                      # 项目说明
├── data/                          # 数据目录
│   ├── BTCUSDT_1h_*.csv          # BTC K线数据
│   ├── BTCUSDT_1h_*.json         # JSON格式数据
│   └── BTCUSDT_1h_*_stats.json   # 数据统计
├── get_btc_data/                 # 数据获取工具
│   ├── get_Data_of_all.py        # 历史数据爬取
│   ├── get_Data_of_realtime.py   # 实时数据监控
│   └── README.md                 # 说明
└── Kronos/                       # Kronos 框架（官方版本）
    ├── finetune_csv/
    │   └── finetuned/
    │       └── BTCUSDT_1h_finetune/
    │           ├── tokenizer/
    │           │   └── best_model/    # ⬇️ 从 HF 下载 Tokenizer
    │           └── basemodel/
    │               └── best_model/    # ⬇️ 从 HF 下载模型
    ├── model/                        # 预训练模型
    ├── examples/                     # 预测示例
    ├── webui/                        # Web界面
    └── 其他官方文件...
```

**重要说明**：`Kronos/` 文件夹包含完整的 Kronos 框架，结构与 [官方仓库](https://github.com/shiyu-coder/Kronos) 保持一致。预测脚本依赖此框架，请勿修改其结构。

## 📊 预测结果示例

系统会自动生成预测结果并保存到本地：

```
预测结果文件:
- btc_prediction_YYYYMMDD_HHMMSS.csv    # CSV 格式预测数据
- btc_prediction_YYYYMMDD_HHMMSS.json   # JSON 格式预测数据
- btc_prediction_YYYYMMDD_HHMMSS.png    # 可视化图表
```

包含内容：
- 历史 BTC/USDT K线数据
- 未来 48 小时的价格预测
- 成交量预测
- 可视化图表（价格 & 成交量）

### 预测结果示意图

![BTC 价格预测示例](btc_prediction_20251020_113426.png)

## 🐛 常见问题

### Q: 模型文件在哪里下载？
A: 模型已上传至 Hugging Face，请从以下链接下载：
- **微调模型**：https://huggingface.co/lc2004/kronos_base_model_BTCUSDT_1h_finetune
- **Tokenizer**：https://huggingface.co/lc2004/kronos_tokenizer_base_BTCUSDT_1h_finetune

### Q: 模型加载失败怎么办？
A: 请确保：
1. 已正确下载模型到指定路径
2. 模型路径配置正确：
   - `./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/tokenizer/best_model`
   - `./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/basemodel/best_model`
3. 文件夹内包含必要的模型文件

### Q: 预测结果保存在哪里？
A: 预测结果默认保存在当前目录，文件名格式：`btc_prediction_YYYYMMDD_HHMMSS.*`

### Q: 可以修改预测时间窗口吗？
A: 可以，修改 `btc_prediction.py` 中的 `pred_len` 参数（默认48小时）。

### Q: 数据获取失败怎么办？
A: 检查网络连接和币安 API 可用性。脚本已内置重试机制，稍等片刻后重试。

## 📝 开发计划

- [x] 微调 BTC 预测模型
- [x] 上传模型到 Hugging Face
- [ ] 支持更多交易对（ETH、BNB等）
- [ ] 添加 Web API 服务
- [ ] 实现回测框架
- [ ] 添加模型评估指标

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 贡献方向
- Bug 报告和问题反馈
- 文档改进
- 性能优化建议
- 新功能需求

## 🙏 致谢

- [Kronos](https://github.com/shiyu-coder/Kronos) - 原始金融预测模型
- [Hugging Face](https://huggingface.co/) - 模型托管平台
- [币安](https://binance.com) - 数据来源
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送微信好友申请至菌子 [+86 15252810681]


## ⚠️ 免责声明

**重要提示**：
- 本项目仅供学习和研究使用
- 预测结果不构成投资建议
- 数字货币交易存在高风险，请谨慎投资
- 作者不对使用本系统造成的任何损失承担责任

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star！**

[Hugging Face 模型](https://huggingface.co/lc2004) | [GitHub 仓库](https://github.com/Liucong-JunZi/Kronos-Btc-finetune)

</div>
