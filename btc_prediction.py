import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加Kronos模块路径
sys.path.append('./Kronos')
from model import Kronos, KronosTokenizer, KronosPredictor

def load_finetuned_models():
    """
    加载微调后的模型和tokenizer
    """
    print("正在加载微调后的模型...")
    
    # 模型路径
    tokenizer_path = "./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/tokenizer/best_model"
    model_path = "./Kronos/finetune_csv/finetuned/BTCUSDT_1h_finetune/basemodel/best_model"
    
    # 加载tokenizer
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer已加载: {tokenizer_path}")
    
    # 加载模型
    model = Kronos.from_pretrained(model_path)
    print(f"模型已加载: {model_path}")
    
    return tokenizer, model

def get_latest_btc_data(days=30):
    """
    获取最新的BTC数据
    """
    print(f"正在获取最近{days}天的BTC数据...")
    
    # 修改get_btc_data脚本的参数
    sys.path.append('./get_btc_data')
    from get_Data_of_all import BinanceHistoricalFetcher
    
    # 创建数据获取器
    fetcher = BinanceHistoricalFetcher(symbol="BTCUSDT", interval="1h")
    
    # 获取数据
    df = fetcher.fetch_historical_data(days=days, save=False)
    
    if df is None or df.empty:
        raise ValueError("无法获取BTC数据")
    
    # 重命名列以匹配模型期望的格式
    df = df.rename(columns={'timestamp': 'timestamps'})
    
    print(f"成功获取{len(df)}条数据记录")
    print(f"数据时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    
    return df

def preprocess_data(df, lookback_window=512):
    """
    预处理数据
    """
    print("正在预处理数据...")
    
    # 确保数据按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    # 添加时间特征
    df['minute'] = df['timestamps'].dt.minute
    df['hour'] = df['timestamps'].dt.hour
    df['weekday'] = df['timestamps'].dt.weekday
    df['day'] = df['timestamps'].dt.day
    df['month'] = df['timestamps'].dt.month
    
    # 检查数据长度
    if len(df) < lookback_window:
        raise ValueError(f"数据长度{len(df)}小于所需的最小长度{lookback_window}")
    
    # 取最后lookback_window条数据作为输入
    input_data = df.iloc[-lookback_window:].copy()
    
    print(f"预处理完成，使用最后{lookback_window}条数据作为输入")
    print(f"输入数据时间范围: {input_data['timestamps'].min()} 到 {input_data['timestamps'].max()}")
    
    return input_data

def generate_future_timestamps(last_timestamp, pred_len, interval='1h'):
    """
    生成未来的时间戳
    """
    future_timestamps = []
    current_time = last_timestamp
    
    for i in range(pred_len):
        if interval == '1h':
            current_time = current_time + timedelta(hours=1)
        elif interval == '1d':
            current_time = current_time + timedelta(days=1)
        # 可以添加更多时间间隔
        
        future_timestamps.append(current_time)
    
    return pd.Series(future_timestamps)

def predict_btc_prices(tokenizer, model, input_data, pred_len=48):
    """
    预测BTC价格
    """
    print(f"正在预测未来{pred_len}小时的BTC价格...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建预测器
    predictor = KronosPredictor(
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        max_context=512, 
        clip=5.0
    )
    
    # 准备输入数据
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    x_df = input_data[feature_cols]
    x_timestamp = input_data['timestamps']
    
    # 生成未来时间戳
    last_timestamp = input_data['timestamps'].iloc[-1]
    y_timestamp = generate_future_timestamps(last_timestamp, pred_len, '1h')
    
    # 进行预测
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=5,  # 使用多个样本进行平均
        verbose=True
    )
    
    # 设置预测结果的时间戳
    pred_df.index = y_timestamp
    
    print("预测完成!")
    return pred_df

def visualize_results(historical_data, predicted_data, lookback_window=100):
    """
    可视化预测结果
    """
    print("正在生成可视化图表...")
    
    # 获取最后lookback_window条历史数据
    recent_history = historical_data.iloc[-lookback_window:].copy()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 价格图表
    ax1.plot(recent_history['timestamps'], recent_history['close'], 
             label='历史价格', color='blue', linewidth=1.5)
    ax1.plot(predicted_data.index, predicted_data['close'], 
             label='预测价格', color='red', linewidth=2, linestyle='--')
    
    ax1.set_title('BTC价格预测', fontsize=16)
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 成交量图表
    ax2.plot(recent_history['timestamps'], recent_history['volume'], 
             label='历史成交量', color='blue', linewidth=1.5)
    ax2.plot(predicted_data.index, predicted_data['volume'], 
             label='预测成交量', color='red', linewidth=2, linestyle='--')
    
    ax2.set_title('BTC成交量预测', fontsize=16)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('时间', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 格式化x轴
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"btc_prediction_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_path}")
    
    plt.show()

def save_prediction_results(historical_data, predicted_data):
    """
    保存预测结果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存预测结果
    pred_path = f"btc_prediction_{timestamp}.csv"
    predicted_data.to_csv(pred_path)
    print(f"预测结果已保存: {pred_path}")
    
    # 保存完整数据（历史+预测）
    combined_data = historical_data.copy()
    
    # 为预测数据创建DataFrame
    pred_df = predicted_data.copy()
    pred_df['timestamps'] = pred_df.index
    
    # 合并数据
    combined_data = pd.concat([combined_data, pred_df], ignore_index=True)
    
    combined_path = f"btc_combined_{timestamp}.csv"
    combined_data.to_csv(combined_path, index=False)
    print(f"合并数据已保存: {combined_path}")

def main():
    """
    主函数
    """
    print("="*60)
    print("BTC价格预测系统")
    print("="*60)
    
    try:
        # 1. 加载微调后的模型
        tokenizer, model = load_finetuned_models()
        
        # 2. 获取最新数据
        df = get_latest_btc_data(days=30)
        
        # 3. 预处理数据
        input_data = preprocess_data(df, lookback_window=512)
        
        # 4. 进行预测
        predicted_data = predict_btc_prices(tokenizer, model, input_data, pred_len=48)
        
        # 5. 显示预测结果
        print("\n预测结果预览:")
        print(predicted_data.head())
        
        # 6. 可视化结果
        visualize_results(df, predicted_data, lookback_window=100)
        
        # 7. 保存结果
        save_prediction_results(df, predicted_data)
        
        print("\n预测完成!")
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()