"""
币安BTC历史数据爬取脚本
功能：获取大量历史K线数据用于模型训练
支持：多时间框架、批量获取、自动保存
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json


class BinanceHistoricalFetcher:
    """币安历史数据爬取器"""
    
    def __init__(self, symbol="BTCUSDT", interval="1h"):
        """
        初始化
        :param symbol: 交易对，如 BTCUSDT, ETHUSDT
        :param interval: K线间隔 (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
        """
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = symbol
        self.interval = interval
        self.data_dir = Path(__file__).parent.parent / "data"
        
    def fetch_klines(self, start_time=None, end_time=None, limit=1000):
        """
        获取K线数据（单次请求）
        :param start_time: 开始时间（毫秒时间戳）
        :param end_time: 结束时间（毫秒时间戳）
        :param limit: 数量限制（最大1000）
        :return: DataFrame
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # 解析数据
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            
            # 添加amount列（成交额）
            df['amount'] = df['quote_volume']
            
            # 只保留需要的列，包含amount
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ 数据解析失败: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, days=90, save=True):
        """
        获取历史数据（支持超过1000条，自动分批）
        :param days: 获取最近N天的数据
        :param save: 是否自动保存
        :return: DataFrame
        """
        print(f"\n{'='*60}")
        print(f"开始获取 {self.symbol} 历史数据")
        print(f"{'='*60}")
        print(f"交易对: {self.symbol}")
        print(f"时间框架: {self.interval}")
        print(f"数据天数: {days} 天")
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        batch_count = 0
        
        print(f"\n开始分批获取数据...")
        
        while current_start < end_time:
            batch_count += 1
            print(f"\r正在获取第 {batch_count} 批数据...", end="", flush=True)
            
            df = self.fetch_klines(
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            
            if df.empty:
                print(f"\n⚠️  第 {batch_count} 批数据为空，停止获取")
                break
            
            all_data.append(df)
            
            # 更新下次开始时间（最后一条数据的时间 + 1毫秒）
            last_timestamp = df['timestamp'].iloc[-1]
            current_start = int(last_timestamp.timestamp() * 1000) + 1
            
            # 避免API限流
            time.sleep(0.5)
            
            # 如果获取的数据少于1000条，说明已经到最新了
            if len(df) < 1000:
                break
        
        print(f"\n\n✓ 共获取 {batch_count} 批数据")
        
        if not all_data:
            print("❌ 未获取到任何数据")
            return None
        
        # 合并所有数据
        result_df = pd.concat(all_data, ignore_index=True)
        
        # 去重和排序
        result_df = result_df.drop_duplicates(subset=['timestamp'])
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ 数据合并完成: {len(result_df)} 条记录")
        print(f"✓ 时间范围: {result_df['timestamp'].min()} ~ {result_df['timestamp'].max()}")
        
        # 保存数据
        if save:
            self.save_data(result_df)
        
        # 显示统计信息
        self.display_statistics(result_df)
        
        return result_df
    
    def save_data(self, df):
        """保存数据到文件"""
        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol}_{self.interval}_{timestamp}"
        
        # 保存为CSV
        csv_path = self.data_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CSV已保存: {csv_path}")
        
        # 保存为JSON（可选）
        json_path = self.data_dir / f"{filename}.json"
        df.to_json(json_path, orient='records', date_format='iso')
        print(f"✓ JSON已保存: {json_path}")
        
        # 保存统计信息
        stats = self.generate_statistics(df)
        stats_path = self.data_dir / f"{filename}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✓ 统计信息已保存: {stats_path}")
    
    def generate_statistics(self, df):
        """生成统计信息"""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "total_records": len(df),
            "start_time": str(df['timestamp'].min()),
            "end_time": str(df['timestamp'].max()),
            "duration_days": (df['timestamp'].max() - df['timestamp'].min()).days,
            "price_stats": {
                "min": float(df['low'].min()),
                "max": float(df['high'].max()),
                "avg": float(df['close'].mean()),
                "first": float(df['open'].iloc[0]),
                "last": float(df['close'].iloc[-1]),
                "change": float(df['close'].iloc[-1] - df['open'].iloc[0]),
                "change_percent": float((df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0] * 100)
            },
            "volume_stats": {
                "total": float(df['volume'].sum()),
                "avg": float(df['volume'].mean()),
                "min": float(df['volume'].min()),
                "max": float(df['volume'].max())
            }
        }
    
    def display_statistics(self, df):
        """显示统计信息"""
        stats = self.generate_statistics(df)
        
        print(f"\n{'='*60}")
        print("📊 数据统计")
        print(f"{'='*60}")
        print(f"交易对: {stats['symbol']}")
        print(f"时间框架: {stats['interval']}")
        print(f"总记录数: {stats['total_records']:,} 条")
        print(f"时间跨度: {stats['duration_days']} 天")
        print(f"\n💰 价格统计:")
        print(f"  最低价: ${stats['price_stats']['min']:,.2f}")
        print(f"  最高价: ${stats['price_stats']['max']:,.2f}")
        print(f"  平均价: ${stats['price_stats']['avg']:,.2f}")
        print(f"  起始价: ${stats['price_stats']['first']:,.2f}")
        print(f"  最新价: ${stats['price_stats']['last']:,.2f}")
        print(f"  涨跌幅: {stats['price_stats']['change_percent']:+.2f}%")
        print(f"\n📈 成交量统计:")
        print(f"  总成交量: {stats['volume_stats']['total']:,.2f} BTC")
        print(f"  平均成交量: {stats['volume_stats']['avg']:,.2f} BTC")
        print(f"  最大成交量: {stats['volume_stats']['max']:,.2f} BTC")
        print(f"{'='*60}")


def main():
    """主函数"""
    print("="*60)
    print("币安BTC历史数据爬取工具")
    print("="*60)
    
    # ========== 配置参数 ==========
    symbol = "BTCUSDT"      # 交易对
    interval = "1h"         # 时间框架 (1m, 5m, 15m, 1h, 4h, 1d)
    days = 730               # 获取最近90天数据
    # ==============================
    
    # 创建爬取器
    fetcher = BinanceHistoricalFetcher(symbol=symbol, interval=interval)
    
    # 获取数据
    df = fetcher.fetch_historical_data(days=days, save=True)
    
    if df is not None and not df.empty:
        print("\n📋 数据预览（前10行）:")
        print(df.head(10).to_string(index=False))
        
        print("\n📋 数据预览（后10行）:")
        print(df.tail(10).to_string(index=False))
        
        print("\n✅ 历史数据获取完成！")
        print(f"数据已保存到: {fetcher.data_dir}")
    else:
        print("\n❌ 数据获取失败")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
