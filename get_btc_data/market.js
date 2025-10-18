const BinanceService = require('./binance');
const config = require('../../config/config');

class MarketService {
    constructor() {
        this.binance = new BinanceService();
        this.ohlcvCache = new Map();
        this.lastUpdate = new Map();
    }

    async getOHLCV(symbol, timeframe = '15m', limit = 200) {
        try {
            const cacheKey = `${symbol}_${timeframe}`;
            const now = Date.now();

            // Check if we have recent data (less than 1 minute old)
            if (this.ohlcvCache.has(cacheKey) &&
                this.lastUpdate.has(cacheKey) &&
                (now - this.lastUpdate.get(cacheKey)) < 60000) {

                return this.ohlcvCache.get(cacheKey);
            }

            // Fetch fresh data - 确保至少获取200根K线用于技术分析
            const fetchLimit = Math.max(limit, 200);
            const ohlcv = await this.binance.exchange.fetchOHLCV(symbol, timeframe, undefined, fetchLimit);

            // Convert to more readable format
            const formattedOHLCV = ohlcv.map(candle => ({
                timestamp: candle[0],
                open: candle[1],
                high: candle[2],
                low: candle[3],
                close: candle[4],
                volume: candle[5] || 0,  // 确保 volume 不是 undefined
                date: new Date(candle[0])
            }));

            // 验证数据完整性
            if (formattedOHLCV.length < fetchLimit * 0.9) {
                console.warn(`Warning: Only got ${formattedOHLCV.length} candles, expected ${fetchLimit}`);
            }

            // Cache the data
            this.ohlcvCache.set(cacheKey, formattedOHLCV);
            this.lastUpdate.set(cacheKey, now);

            return formattedOHLCV;

        } catch (error) {
            console.error(`Failed to get OHLCV data for ${symbol} ${timeframe}:`, error.message);

            // 如果缓存中有数据，即使过期也返回
            if (this.ohlcvCache.has(cacheKey)) {
                console.warn(`Using cached data for ${symbol} ${timeframe} due to error`);
                return this.ohlcvCache.get(cacheKey);
            }

            throw error;
        }
    }

    async getMultipleTimeframes(symbol) {
        try {
            const timeframes = config.trading.timeframes;
            const results = {};

            // 为每个时间框架获取足够的K线数据（至少200根）
            for (const [key, timeframe] of Object.entries(timeframes)) {
                results[key] = await this.getOHLCV(symbol, timeframe, 200);
            }

            return results;

        } catch (error) {
            console.error(`Failed to get multiple timeframes for ${symbol}:`, error.message);
            throw error;
        }
    }

    async getLatestCandle(symbol, timeframe = '15m') {
        try {
            const candles = await this.getOHLCV(symbol, timeframe, 1);
            return candles.length > 0 ? candles[0] : null;

        } catch (error) {
            console.error(`Failed to get latest candle for ${symbol} ${timeframe}:`, error.message);
            throw error;
        }
    }

    async getVolumeData(symbol, timeframe = '15m', periods = 20) {
        try {
            const candles = await this.getOHLCV(symbol, timeframe, periods);

            const volumes = candles.map(candle => candle.volume);
            const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;

            return {
                currentVolume: volumes[volumes.length - 1],
                avgVolume: avgVolume,
                volumeRatio: volumes[volumes.length - 1] / avgVolume,
                volumes: volumes,
            };

        } catch (error) {
            console.error(`Failed to get volume data for ${symbol}:`, error.message);
            throw error;
        }
    }

    async getFundingRate(symbol) {
        try {
            const fundingRate = await this.binance.exchange.fetchFundingRate(symbol);
            return {
                symbol: fundingRate.symbol,
                fundingRate: fundingRate.fundingRate,
                fundingTimestamp: fundingRate.fundingTimestamp,
                timestamp: fundingRate.timestamp,
            };

        } catch (error) {
            console.error(`Failed to get funding rate for ${symbol}:`, error.message);
            throw error;
        }
    }

    async getOpenInterest(symbol) {
        try {
            const openInterest = await this.binance.exchange.fetchOpenInterest(symbol);
            return {
                symbol: openInterest.symbol,
                openInterest: openInterest.openInterest,
                timestamp: openInterest.timestamp,
            };

        } catch (error) {
            console.error(`Failed to get open interest for ${symbol}:`, error.message);
            throw error;
        }
    }

    async getMarketData(symbol) {
        try {
            const [ohlcv, volumeData, fundingRate, openInterest] = await Promise.all([
                this.getMultipleTimeframes(symbol),
                this.getVolumeData(symbol, config.trading.timeframes.primary),
                this.getFundingRate(symbol),
                this.getOpenInterest(symbol),
            ]);

            return {
                symbol,
                ohlcv,
                volume: volumeData,
                funding: fundingRate,
                openInterest: openInterest,
                timestamp: Date.now(),
            };

        } catch (error) {
            console.error(`Failed to get comprehensive market data for ${symbol}:`, error.message);
            throw error;
        }
    }

    clearCache() {
        this.ohlcvCache.clear();
        this.lastUpdate.clear();
    }
}

module.exports = MarketService;