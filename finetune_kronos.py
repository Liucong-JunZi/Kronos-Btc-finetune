"""
Finetune the Kronos-base model on custom BTC data.

This script performs a true finetuning of the Kronos model by:
1. Loading the pre-trained Kronos-base model and its corresponding tokenizer.
2. Processing a custom time-series dataset (e.g., BTC prices).
3. Using the KronosTokenizer to convert the continuous OHLCV data into discrete tokens.
4. Setting up a PyTorch Dataset and DataLoader for the tokenized data.
5. Running a training loop to finetune the model on the new data.
6. Saving the finetuned model weights.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

# ==============================================================================
#  强制将 'Kronos' 目录添加到 Python 路径 (关键修复)
# ==============================================================================
try:
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent.resolve()
    # 构建 'Kronos' 目录的完整路径
    kronos_path = current_dir / 'Kronos'
    
    # 检查 'Kronos' 目录是否存在
    if not kronos_path.is_dir():
        raise FileNotFoundError(f"错误: 'Kronos' 目录未在 {current_dir} 中找到。")
    
    # 将 'Kronos' 目录的父目录添加到 sys.path
    # 这样我们就可以使用 'from Kronos.model import ...'
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 现在可以安全地导入
    from Kronos.model import Kronos, KronosTokenizer
    print("成功从 'Kronos' 目录导入模型定义。")

except (ImportError, FileNotFoundError) as e:
    print("\n" + "="*60)
    print("致命错误: 无法导入 Kronos 模型定义。")
    print(f"   请确保您已将官方的 'Kronos' GitHub 仓库克隆到项目根目录: {current_dir}")
    print(f"   原始错误: {e}")
    print("="*60)
    sys.exit(1) # 导入失败，直接退出
# ==============================================================================

# --- Configuration ---
# Paths
DATA_FILE = "data/BTCUSDT_1h_20251018_220012.csv"
BASE_MODEL_DIR = "Kronos-base"
OUTPUT_DIR = "output"
FINETUNED_MODEL_PATH = os.path.join(OUTPUT_DIR, "finetuned_kronos_btc.safetensors")
TOKENIZER_CACHE_DIR = "kronos_tokenizer_cache" # Directory to cache the downloaded tokenizer

# Model & Tokenizer
BASE_MODEL_CONFIG = os.path.join(BASE_MODEL_DIR, "config.json")
BASE_MODEL_WEIGHTS = os.path.join(BASE_MODEL_DIR, "model.safetensors")
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

# Finetuning Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 10
CONTEXT_LENGTH = 512  # Max context length for Kronos-base
TRAIN_SPLIT = 0.9 # 90% for training, 10% for validation

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """主微调函数"""
    # ========== 配置 ==========
    config = {
        "data_path": "data/BTCUSDT_1h_20251018_220012.csv",
        "base_model_path": "Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "output_model_path": "output/finetuned_kronos_btc.pth",
        "log_path": "output/training_log.json",
        "lookback_window": 512,
        "predict_window": 48,
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("\n" + "="*60)
    print("KRONOS 真实微调流程")
    print("="*60)
    for key, value in config.items():
        print(f"  - {key:<20}: {value}")
    print("="*60 + "\n")

    # 1. 加载数据
    print("Step 1: 正在加载并预处理数据...")
    df = pd.read_csv(config["data_path"])
    # 只保留模型需要的6列数据
    data_for_tokenizer = df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    print(f"数据加载成功，共 {len(data_for_tokenizer)} 条记录。")

    # 2. 加载模型和分词器
    print("\nStep 2: 正在加载预训练模型和分词器...")
    device = torch.device(config["device"])
    
    # 加载分词器
    tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_id"])
    print(f"分词器 '{config['tokenizer_id']}' 加载成功。")

    # 加载模型
    model = Kronos.from_pretrained(config["base_model_path"]).to(device)
    print(f"预训练模型 '{config['base_model_path']}' 加载成功。")
    print(f"   模型总参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 3. 数据分词 (Tokenization)
    print("\nStep 3: 正在将K线数据转换为金融词元 (Tokens)...")
    # 将DataFrame转换为torch tensor，确保形状为 (seq_len, 6)
    ts_data = torch.tensor(data_for_tokenizer.values, dtype=torch.float32)
    print(f"输入数据形状: {ts_data.shape}")
    
    # 重塑数据为 (1, seq_len, 6) 以匹配分词器期望的输入格式
    ts_data = ts_data.unsqueeze(0)  # (1, seq_len, 6)
    print(f"重塑后数据形状: {ts_data.shape}")
    
    # 使用分词器进行编码，启用分层分词
    token_seq_0, token_seq_1 = tokenizer.encode(ts_data, half=True)  # 输入: (1, seq_len, 6)
    print(f"数据分词成功，生成分层词元。")
    print(f"S1 tokens形状: {token_seq_0.shape}")
    print(f"S2 tokens形状: {token_seq_1.shape}")

    # 4. 创建数据集和数据加载器
    print("\nStep 4: 正在创建时间序列训练数据集...")
    
    # 严格按时间顺序分割数据 - 修复数据泄露问题
    total_len = config["lookback_window"] + config["predict_window"]
    
    # 处理分层词元 - 取第一个batch
    s1_tokens = token_seq_0[0]  # (seq_len,)
    s2_tokens = token_seq_1[0]  # (seq_len,)
    
    print(f"S1 tokens形状: {s1_tokens.shape}")
    print(f"S2 tokens形状: {s2_tokens.shape}")
    print(f"分词后数据长度: {len(s1_tokens)}")
    print(f"需要的总长度: {total_len}")
    
    # 严格的时间序列分割 - 修复数据泄露问题
    train_ratio = 0.7
    val_ratio = 0.15
    train_end = int(len(s1_tokens) * train_ratio)
    val_end = int(len(s1_tokens) * (train_ratio + val_ratio))
    
    print(f"数据分割: 训练集0-{train_end}, 验证集{train_end}-{val_end}, 测试集{val_end}-{len(s1_tokens)}")
    
    def create_time_series_samples(tokens, start_idx, end_idx, lookback_window, predict_window):
        """创建时间序列样本，避免数据泄露"""
        samples = []
        total_len = lookback_window + predict_window
        
        for i in range(start_idx, end_idx - total_len + 1):
            # 创建严格的时间序列样本 - 输入只包含历史数据，目标包含未来数据
            sample = tokens[i : i + total_len]
            samples.append(sample)
        
        return samples
    
    # 创建不重叠的训练、验证、测试样本
    train_samples_s1 = create_time_series_samples(s1_tokens, 0, train_end, config["lookback_window"], config["predict_window"])
    val_samples_s1 = create_time_series_samples(s1_tokens, train_end, val_end, config["lookback_window"], config["predict_window"])
    test_samples_s1 = create_time_series_samples(s1_tokens, val_end, len(s1_tokens), config["lookback_window"], config["predict_window"])
    
    train_samples_s2 = create_time_series_samples(s2_tokens, 0, train_end, config["lookback_window"], config["predict_window"])
    val_samples_s2 = create_time_series_samples(s2_tokens, train_end, val_end, config["lookback_window"], config["predict_window"])
    test_samples_s2 = create_time_series_samples(s2_tokens, val_end, len(s2_tokens), config["lookback_window"], config["predict_window"])
    
    print(f"训练样本数: {len(train_samples_s1)}")
    print(f"验证样本数: {len(val_samples_s1)}")
    print(f"测试样本数: {len(test_samples_s1)}")
    
    # 创建训练数据集
    X_train_s1 = torch.stack(train_samples_s1)
    X_train_s2 = torch.stack(train_samples_s2)
    
    train_dataset = TensorDataset(X_train_s1, X_train_s2)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    print(f"训练数据集创建成功，共 {len(train_dataset)} 个样本。")

    # 5. 微调训练
    print("\nStep 5: 开始微调训练...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    training_log = []
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_X_s1, batch_X_s2 in progress_bar:
            batch_X_s1, batch_X_s2 = batch_X_s1.to(device), batch_X_s2.to(device)
            
            optimizer.zero_grad()
            
            # 准备输入和目标 (按照官方示例的方式)
            token_in_s1 = batch_X_s1[:, :-1]  # 输入序列 (去掉最后一个)
            token_in_s2 = batch_X_s2[:, :-1]  # 输入序列 (去掉最后一个)
            token_out_s1 = batch_X_s1[:, 1:]  # 目标序列 (去掉第一个)
            token_out_s2 = batch_X_s2[:, 1:]  # 目标序列 (去掉第一个)
            
            # 获取模型输出
            s1_logits, s2_logits = model(token_in_s1, token_in_s2) # (batch, seq_len-1, vocab_size)
            
            # 使用模型的head计算损失 (按照官方示例)
            loss, s1_loss, s2_loss = model.head.compute_loss(s1_logits, s2_logits, token_out_s1, token_out_s2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.6f}")
        training_log.append({"epoch": epoch+1, "avg_loss": avg_loss})

    print("\n微调训练完成！")

    # 6. 验证模型性能
    print("\nStep 6: 验证模型性能...")
    if len(val_samples_s1) > 0:
        X_val_s1 = torch.stack(val_samples_s1)
        X_val_s2 = torch.stack(val_samples_s2)
        val_dataset = TensorDataset(X_val_s1, X_val_s2)
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X_s1, batch_X_s2 in val_dataloader:
                batch_X_s1, batch_X_s2 = batch_X_s1.to(device), batch_X_s2.to(device)
                
                token_in_s1 = batch_X_s1[:, :-1]
                token_in_s2 = batch_X_s2[:, :-1]
                token_out_s1 = batch_X_s1[:, 1:]
                token_out_s2 = batch_X_s2[:, 1:]
                
                s1_logits, s2_logits = model(token_in_s1, token_in_s2)
                loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, token_out_s1, token_out_s2)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"验证损失: {avg_val_loss:.4f}")
        training_log.append({"epoch": "validation", "avg_loss": avg_val_loss})
    else:
        print("警告: 验证集为空，跳过验证")

    # 7. 保存模型和日志
    print("\nStep 7: 正在保存微调后的模型和日志...")
    # 确保输出目录存在
    Path("output").mkdir(exist_ok=True)
    
    # 保存模型状态字典
    torch.save(model.state_dict(), config["output_model_path"])
    print(f"模型已保存到: {config['output_model_path']}")
    
    # 保存日志
    with open(config["log_path"], 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"训练日志已保存到: {config['log_path']}")
    
    print("\n" + "="*60)
    print("KRONOS 微调完成！")
    print("="*60)
    print(f"最终训练损失: {training_log[-2]['avg_loss']:.4f}")
    if len(val_samples_s1) > 0:
        print(f"验证损失: {training_log[-1]['avg_loss']:.4f}")
    print(f"模型保存路径: {config['output_model_path']}")
    print("="*60)

if __name__ == "__main__":
    main()
