"""
币安BTC实时数据爬取脚本
功能：持续获取最新K线数据并追加保存
支持：实时更新、自动追加、异常恢复
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import signal
import sys


class BinanceRealtimeFetcher:
    """币安实时数据爬取器"""
    
    def __init__(self, symbol="BTCUSDT", interval="1h", update_interval=60):
        """
        初始化
        :param symbol: 交易对
        :param interval: K线间隔
        :param update_interval: 更新间隔（秒）
        """
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
        self.symbol = symbol
        self.interval = interval
        self.update_interval = update_interval
        self.data_dir = Path(__file__).parent.parent / "data" / "realtime"
        self.running = True
        
        # 设置信号处理（优雅退出）
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """信号处理器"""
        print("\n\n⚠️  接收到停止信号，正在安全退出...")
        self.running = False
    
    def fetch_latest_kline(self, limit=1):
        """
        获取最新的K线数据
        :param limit: 获取数量
        :return: DataFrame
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        }
        
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
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # 添加额外信息
            df['fetch_time'] = datetime.now()
            
            # 只保留需要的列
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'fetch_time']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ 数据解析失败: {e}")
            return pd.DataFrame()
    
    def fetch_ticker(self):
        """获取24小时行情数据"""
        try:
            response = requests.get(self.ticker_url, params={"symbol": self.symbol}, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "price": float(data['lastPrice']),
                "change": float(data['priceChange']),
                "change_percent": float(data['priceChangePercent']),
                "high_24h": float(data['highPrice']),
                "low_24h": float(data['lowPrice']),
                "volume_24h": float(data['volume']),
                "quote_volume_24h": float(data['quoteVolume']),
                "trades_24h": int(data['count'])
            }
            
        except Exception as e:
            print(f"❌ 获取行情失败: {e}")
            return None
    
    def initialize_file(self):
        """初始化数据文件"""
        # 创建目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        date_str = datetime.now().strftime("%Y%m%d")
        self.csv_file = self.data_dir / f"{self.symbol}_{self.interval}_{date_str}.csv"
        self.json_file = self.data_dir / f"{self.symbol}_{self.interval}_latest.json"
        
        # 如果CSV文件不存在，创建表头
        if not self.csv_file.exists():
            with open(self.csv_file, 'w') as f:
                f.write("timestamp,open,high,low,close,volume,fetch_time\n")
            print(f"✓ 创建新文件: {self.csv_file}")
        else:
            print(f"✓ 追加到现有文件: {self.csv_file}")
    
    def append_data(self, df):
        """追加数据到CSV文件"""
        if df.empty:
            return False
        
        try:
            # 追加到CSV
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
            
            # 保存最新数据为JSON（覆盖）
            latest_data = df.iloc[-1].to_dict()
            latest_data['timestamp'] = str(latest_data['timestamp'])
            latest_data['fetch_time'] = str(latest_data['fetch_time'])
            
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(latest_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False
    
    def display_realtime_info(self, kline_data, ticker_data):
        """显示实时信息"""
        if kline_data.empty:
            return
        
        latest = kline_data.iloc[-1]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*70}")
        print(f"⏰ 更新时间: {now}")
        print(f"{'='*70}")
        print(f"📊 K线数据 ({self.interval}):")
        print(f"  时间: {latest['timestamp']}")
        print(f"  开盘: ${latest['open']:,.2f}")
        print(f"  最高: ${latest['high']:,.2f}")
        print(f"  最低: ${latest['low']:,.2f}")
        print(f"  收盘: ${latest['close']:,.2f}")
        print(f"  成交量: {latest['volume']:,.2f} BTC")
        
        if ticker_data:
            print(f"\n💹 24小时行情:")
            print(f"  当前价: ${ticker_data['price']:,.2f}")
            print(f"  涨跌: {ticker_data['change']:+,.2f} ({ticker_data['change_percent']:+.2f}%)")
            print(f"  24H最高: ${ticker_data['high_24h']:,.2f}")
            print(f"  24H最低: ${ticker_data['low_24h']:,.2f}")
            print(f"  24H成交量: {ticker_data['volume_24h']:,.2f} BTC")
            print(f"  24H成交额: ${ticker_data['quote_volume_24h']:,.0f}")
            print(f"  24H交易笔数: {ticker_data['trades_24h']:,}")
        
        print(f"{'='*70}")
        print(f"📁 数据文件: {self.csv_file.name}")
        print(f"⏳ 下次更新: {self.update_interval} 秒后")
    
    def run(self, backfill_hours=24):
        """
        运行实时数据获取
        :param backfill_hours: 启动时回填多少小时的历史数据
        """
        print(f"\n{'='*70}")
        print(f"币安实时数据爬取器")
        print(f"{'='*70}")
        print(f"交易对: {self.symbol}")
        print(f"时间框架: {self.interval}")
        print(f"更新间隔: {self.update_interval} 秒")
        print(f"{'='*70}")
        
        # 初始化文件
        self.initialize_file()
        
        # 回填历史数据
        if backfill_hours > 0:
            print(f"\n正在回填最近 {backfill_hours} 小时的数据...")
            
            limit = backfill_hours if self.interval == "1h" else int(backfill_hours * 60 / int(self.interval[:-1]))
            historical_data = self.fetch_latest_kline(limit=min(limit, 1000))
            
            if not historical_data.empty:
                self.append_data(historical_data)
                print(f"✓ 已回填 {len(historical_data)} 条历史数据")
        
        print(f"\n✓ 启动完成，开始实时监控...")
        print(f"提示: 按 Ctrl+C 可安全退出\n")
        
        # 记录上次的时间戳，避免重复
        last_timestamp = None
        error_count = 0
        max_errors = 5
        
        while self.running:
            try:
                # 获取最新K线
                kline_data = self.fetch_latest_kline(limit=1)
                
                if not kline_data.empty:
                    current_timestamp = kline_data.iloc[0]['timestamp']
                    
                    # 只保存新数据（避免重复）
                    if last_timestamp is None or current_timestamp != last_timestamp:
                        if self.append_data(kline_data):
                            last_timestamp = current_timestamp
                            error_count = 0  # 重置错误计数
                    
                    # 获取并显示行情数据
                    ticker_data = self.fetch_ticker()
                    self.display_realtime_info(kline_data, ticker_data)
                else:
                    print(f"⚠️  [{datetime.now().strftime('%H:%M:%S')}] 未获取到数据")
                    error_count += 1
                
                # 如果连续错误太多，退出
                if error_count >= max_errors:
                    print(f"\n❌ 连续 {max_errors} 次错误，程序退出")
                    break
                
                # 等待下次更新
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"\n❌ 运行异常: {e}")
                error_count += 1
                
                if error_count >= max_errors:
                    print(f"❌ 错误次数过多，程序退出")
                    break
                
                print(f"⏳ {self.update_interval} 秒后重试...")
                time.sleep(self.update_interval)
        
        print(f"\n✓ 程序已安全退出")
        print(f"✓ 数据已保存到: {self.csv_file}")


def main():
    """主函数"""
    # ========== 配置参数 ==========
    symbol = "BTCUSDT"          # 交易对
    interval = "1h"             # 时间框架 (1m, 5m, 15m, 1h, 4h)
    update_interval = 60        # 更新间隔（秒）
    backfill_hours = 24         # 启动时回填24小时数据
    # ==============================
    
    # 创建实时爬取器
    fetcher = BinanceRealtimeFetcher(
        symbol=symbol,
        interval=interval,
        update_interval=update_interval
    )
    
    # 运行
    fetcher.run(backfill_hours=backfill_hours)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
