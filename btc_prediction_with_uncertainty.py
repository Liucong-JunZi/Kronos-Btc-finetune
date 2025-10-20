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

def predict_btc_prices_with_uncertainty(tokenizer, model, input_data, pred_len=48, sample_count=20):
    """
    预测BTC价格并计算不确定性
    """
    print(f"正在预测未来{pred_len}小时的BTC价格，使用{sample_count}次采样计算不确定性...")
    
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
    
    # 进行多次预测以计算不确定性
    all_predictions = []
    
    for i in range(sample_count):
        print(f"正在进行第{i+1}/{sample_count}次采样...")
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,  # 温度参数，控制随机性
            top_p=0.9,
            sample_count=1,  # 每次只采样一次
            verbose=False
        )
        
        all_predictions.append(pred_df)
    
    # 合并所有预测结果
    print("正在计算预测统计信息...")
    
    # 将所有预测结果转换为numpy数组
    pred_arrays = []
    for pred in all_predictions:
        pred_arrays.append(pred.values)
    
    pred_stack = np.stack(pred_arrays, axis=0)  # shape: (sample_count, pred_len, n_features)
    
    # 计算统计量
    pred_mean = np.mean(pred_stack, axis=0)
    pred_std = np.std(pred_stack, axis=0)
    pred_min = np.min(pred_stack, axis=0)
    pred_max = np.max(pred_stack, axis=0)
    
    # 计算置信区间 (95%)
    percentiles_5 = np.percentile(pred_stack, 5, axis=0)
    percentiles_95 = np.percentile(pred_stack, 95, axis=0)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(pred_mean, columns=feature_cols, index=y_timestamp)
    
    # 添加不确定性度量
    for col in feature_cols:
        result_df[f'{col}_std'] = pred_std[:, feature_cols.index(col)]
        result_df[f'{col}_min'] = pred_min[:, feature_cols.index(col)]
        result_df[f'{col}_max'] = pred_max[:, feature_cols.index(col)]
        result_df[f'{col}_5th'] = percentiles_5[:, feature_cols.index(col)]
        result_df[f'{col}_95th'] = percentiles_95[:, feature_cols.index(col)]
    
    print("预测完成!")
    return result_df, all_predictions

def print_prediction_statistics(result_df, feature_cols):
    """
    打印预测统计信息
    """
    print("\n" + "="*60)
    print("预测统计信息")
    print("="*60)
    
    for col in feature_cols:
        print(f"\n{col.upper()} 预测统计:")
        print(f"  平均值: {result_df[col].mean():.4f}")
        print(f"  标准差: {result_df[f'{col}_std'].mean():.4f}")
        print(f"  最小值: {result_df[col].min():.4f}")
        print(f"  最大值: {result_df[col].max():.4f}")
        print(f"  变异系数: {(result_df[f'{col}_std'].mean() / result_df[col].mean()):.4f}")
        
        # 计算价格变化（如果是价格相关列）
        if col in ['open', 'high', 'low', 'close']:
            price_change = result_df[col].iloc[-1] - result_df[col].iloc[0]
            price_change_pct = (price_change / result_df[col].iloc[0]) * 100
            print(f"  预测期间价格变化: {price_change:+.4f} ({price_change_pct:+.2f}%)")

def visualize_results_with_uncertainty(historical_data, result_df, feature_cols, lookback_window=100):
    """
    可视化带不确定性的预测结果
    """
    print("正在生成带不确定性的可视化图表...")
    
    # 获取最后lookback_window条历史数据
    recent_history = historical_data.iloc[-lookback_window:].copy()
    
    # 创建图表
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(12, 4*len(feature_cols)))
    if len(feature_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        
        # 绘制历史数据
        ax.plot(recent_history['timestamps'], recent_history[col], 
                label='历史数据', color='blue', linewidth=1.5, alpha=0.7)
        
        # 绘制预测均值
        ax.plot(result_df.index, result_df[col], 
                label='预测均值', color='red', linewidth=2, linestyle='--')
        
        # 绘制95%置信区间
        ax.fill_between(result_df.index, 
                       result_df[f'{col}_5th'], 
                       result_df[f'{col}_95th'], 
                       alpha=0.3, color='red', label='95%置信区间')
        
        # 绘制标准差范围
        ax.fill_between(result_df.index, 
                       result_df[col] - result_df[f'{col}_std'], 
                       result_df[col] + result_df[f'{col}_std'], 
                       alpha=0.2, color='orange', label='±1标准差')
        
        ax.set_title(f'{col.upper()} 预测 (带不确定性)', fontsize=14)
        ax.set_ylabel(col.upper(), fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"btc_prediction_uncertainty_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"不确定性图表已保存: {plot_path}")
    
    plt.show()

def save_prediction_results_with_uncertainty(historical_data, result_df, all_predictions):
    """
    保存带不确定性的预测结果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存主要预测结果
    pred_path = f"btc_prediction_uncertainty_{timestamp}.csv"
    result_df.to_csv(pred_path)
    print(f"带不确定性的预测结果已保存: {pred_path}")
    
    # 保存所有采样结果
    all_preds_path = f"btc_all_samples_{timestamp}.csv"
    
    # 合并所有采样结果
    sample_dfs = []
    for i, pred in enumerate(all_predictions):
        pred_copy = pred.copy()
        pred_copy.columns = [f"{col}_sample_{i}" for col in pred.columns]
        sample_dfs.append(pred_copy)
    
    all_samples_df = pd.concat(sample_dfs, axis=1)
    all_samples_df.to_csv(all_preds_path)
    print(f"所有采样结果已保存: {all_preds_path}")

def main():
    """
    主函数
    """
    print("="*60)
    print("BTC价格预测系统 (带不确定性分析)")
    print("="*60)
    
    try:
        # 1. 加载微调后的模型
        tokenizer, model = load_finetuned_models()
        
        # 2. 获取最新数据
        df = get_latest_btc_data(days=30)
        
        # 3. 预处理数据
        input_data = preprocess_data(df, lookback_window=512)
        
        # 4. 进行预测（带不确定性）
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        result_df, all_predictions = predict_btc_prices_with_uncertainty(
            tokenizer, model, input_data, pred_len=48, sample_count=20
        )
        
        # 5. 显示预测统计信息
        print_prediction_statistics(result_df, feature_cols)
        
        # 6. 显示预测结果预览
        print("\n预测结果预览 (前5行):")
        display_cols = feature_cols + [f'{col}_std' for col in feature_cols]
        print(result_df[display_cols].head())
        
        # 7. 可视化结果
        visualize_results_with_uncertainty(df, result_df, feature_cols, lookback_window=100)
        
        # 8. 保存结果
        save_prediction_results_with_uncertainty(df, result_df, all_predictions)
        
        print("\n预测完成!")
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()