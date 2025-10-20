import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
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

def visualize_kline_with_probability(historical_data, result_df, lookback_window=100):
    """
    使用K线图形式可视化预测结果，带概率分布
    """
    print("正在生成K线图形式的可视化...")
    
    # 获取最后lookback_window条历史数据
    recent_history = historical_data.iloc[-lookback_window:].copy()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # 设置颜色
    bullish_color = '#26a69a'  # 绿色（上涨）
    bearish_color = '#ef5350'  # 红色（下跌）
    
    # 1. 绘制历史K线图
    for i, (idx, row) in enumerate(recent_history.iterrows()):
        color = bullish_color if row['close'] >= row['open'] else bearish_color
        
        # 绘制K线实体
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['close'], row['open'])
        
        # 实体
        ax1.add_patch(Rectangle(
            (float(mdates.date2num(row['timestamps'])) - 0.0003, body_bottom),
            0.0006, body_height,
            color=color, alpha=0.8
        ))
        
        # 影线
        ax1.plot([mdates.date2num(row['timestamps']), mdates.date2num(row['timestamps'])],
                [row['low'], row['high']], 
                color=color, linewidth=1, alpha=0.8)
    
    # 2. 绘制预测K线图（带概率）
    for i, (idx, row) in enumerate(result_df.iterrows()):
        # 计算预测的涨跌概率
        prob_up = (row['close'] > row['open']).astype(float)
        
        # 根据概率设置颜色透明度
        alpha = 0.3 + 0.5 * abs(prob_up - 0.5) * 2  # 概率越确定，透明度越高
        
        # 使用预测的5th和95th分位数作为不确定性的范围
        low_5th = row['low_5th']
        high_95th = row['high_95th']
        
        # 绘制不确定性范围（背景）
        ax1.fill_between([mdates.date2num(idx) - 0.0004, mdates.date2num(idx) + 0.0004],
                        low_5th, high_95th,
                        color='gray', alpha=0.2, label='不确定性范围' if i == 0 else "")
        
        # 绘制预测K线
        color = bullish_color if row['close'] >= row['open'] else bearish_color
        
        # 实体
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['close'], row['open'])
        
        ax1.add_patch(Rectangle(
            (float(mdates.date2num(idx)) - 0.0003, body_bottom),
            0.0006, body_height,
            color=color, alpha=alpha, label='预测K线' if i == 0 else ""
        ))
        
        # 影线
        ax1.plot([mdates.date2num(idx), mdates.date2num(idx)],
                [row['low'], row['high']], 
                color=color, linewidth=1, alpha=alpha)
    
    # 3. 添加概率热力图背景
    # 计算每个预测时间点的整体不确定性
    uncertainty_scores = []
    for idx, row in result_df.iterrows():
        # 计算价格范围相对于均值的比例作为不确定性度量
        price_uncertainty = (row['high_95th'] - row['low_5th']) / row['close']
        uncertainty_scores.append(price_uncertainty)
    
    # 在底部添加不确定性热力图
    ax3 = ax1.twinx()
    ax3.bar(result_df.index, uncertainty_scores, 
            width=pd.Timedelta(hours=0.8), 
            alpha=0.3, color='orange', label='预测不确定性')
    ax3.set_ylabel('不确定性程度', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3.set_ylim(0, max(uncertainty_scores) * 1.2 if uncertainty_scores else 0.1)
    
    # 设置主图表
    ax1.set_title('BTC价格预测 - K线图 (带概率分布)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 格式化x轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper left')
    
    # 4. 绘制成交量图
    # 历史成交量
    ax2.bar(recent_history['timestamps'], recent_history['volume'], 
            width=pd.Timedelta(hours=0.8), 
            color='blue', alpha=0.6, label='历史成交量')
    
    # 预测成交量（带不确定性）
    for idx, row in result_df.iterrows():
        vol_mean = row['volume']
        vol_std = row['volume_std']
        
        # 绘制成交量范围
        ax2.bar(idx, vol_mean, 
                width=pd.Timedelta(hours=0.8), 
                color='red', alpha=0.6, label='预测成交量' if idx == result_df.index[0] else "")
        
        # 添加不确定性范围
        ax2.errorbar(idx, vol_mean, yerr=vol_std, 
                    fmt='none', ecolor='red', alpha=0.3, capsize=3)
    
    ax2.set_title('成交量预测', fontsize=14)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加垂直线分隔历史和预测
    if len(recent_history) > 0 and len(result_df) > 0:
        last_hist_time = recent_history['timestamps'].iloc[-1]
        ax1.axvline(x=last_hist_time, color='black', linestyle='--', alpha=0.5, linewidth=2)
        ax2.axvline(x=last_hist_time, color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        # 添加文字标注
        ax1.text(last_hist_time - pd.Timedelta(hours=12), ax1.get_ylim()[1] * 0.95, 
                '历史数据', ha='right', fontsize=12, fontweight='bold')
        ax1.text(last_hist_time + pd.Timedelta(hours=12), ax1.get_ylim()[1] * 0.95, 
                '预测数据', ha='left', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"btc_prediction_kline_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"K线图已保存: {plot_path}")
    
    plt.show()

def print_prediction_summary(result_df):
    """
    打印预测摘要信息
    """
    print("\n" + "="*60)
    print("预测摘要")
    print("="*60)
    
    # 计算整体趋势
    price_change = result_df['close'].iloc[-1] - result_df['close'].iloc[0]
    price_change_pct = (price_change / result_df['close'].iloc[0]) * 100
    
    trend = "上涨" if price_change > 0 else "下跌" if price_change < 0 else "横盘"
    
    print(f"预测期间: {result_df.index[0]} 到 {result_df.index[-1]}")
    print(f"预测趋势: {trend}")
    print(f"价格变化: {price_change:+.2f} USDT ({price_change_pct:+.2f}%)")
    print(f"起始价格: {result_df['close'].iloc[0]:.2f} USDT")
    print(f"结束价格: {result_df['close'].iloc[-1]:.2f} USDT")
    print(f"最高预测价: {result_df['high'].max():.2f} USDT")
    print(f"最低预测价: {result_df['low'].min():.2f} USDT")
    
    # 计算平均不确定性
    avg_uncertainty = np.mean([(row['high_95th'] - row['low_5th']) / row['close'] 
                              for _, row in result_df.iterrows()])
    print(f"平均预测不确定性: {avg_uncertainty:.2%}")
    
    # 计算上涨概率
    up_periods = sum(1 for _, row in result_df.iterrows() if row['close'] > row['open'])
    up_probability = up_periods / len(result_df)
    print(f"预测期间上涨概率: {up_probability:.2%}")

def main():
    """
    主函数
    """
    print("="*60)
    print("BTC价格预测系统 - K线图版本")
    print("="*60)
    
    try:
        # 1. 加载微调后的模型
        tokenizer, model = load_finetuned_models()
        
        # 2. 获取最新数据
        df = get_latest_btc_data(days=30)
        
        # 3. 预处理数据
        input_data = preprocess_data(df, lookback_window=512)
        
        # 4. 进行预测（带不确定性）
        result_df, all_predictions = predict_btc_prices_with_uncertainty(
            tokenizer, model, input_data, pred_len=48, sample_count=20
        )
        
        # 5. 打印预测摘要
        print_prediction_summary(result_df)
        
        # 6. 可视化结果（K线图）
        visualize_kline_with_probability(df, result_df, lookback_window=100)
        
        # 7. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = f"btc_prediction_kline_{timestamp}.csv"
        result_df.to_csv(pred_path)
        print(f"\n预测结果已保存: {pred_path}")
        
        print("\n预测完成!")
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()