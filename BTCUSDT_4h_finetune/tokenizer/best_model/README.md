---
language:
- en
- zh
license: mit
tags:
- tokenizer
- time-series
- bitcoin
- btc
- cryptocurrency
- numeric-encoding
---

# BTCUSDT 4-Hour Tokenizer

## Tokenizer Description

This is a specialized tokenizer designed for **time-series cryptocurrency data encoding**, specifically fine-tuned for BTCUSDT (Bitcoin/USDT) 4-hour candlestick data. It converts numerical trading data (OHLCV - Open, High, Low, Close, Volume) into token representations suitable for transformer-based models.

### Tokenizer Details

- **Type**: Numeric Time-Series Tokenizer
- **Vocabulary Size**: Model-specific
- **Input Format**: BTCUSDT candlestick data (OHLCV)
- **Output**: Token sequences for model inference
- **Framework**: Hugging Face Transformers compatible
- **Time Granularity**: 4-hour candlesticks

## Purpose

This tokenizer is used to preprocess historical BTCUSDT 4-hour trading data before feeding it into the fine-tuned prediction model. It handles:

- **Price normalization**: Converts raw price values to a standardized token space
- **Volume encoding**: Encodes trading volume information
- **Temporal sequences**: Preserves time-series relationships in medium-term data
- **Model compatibility**: Ensures proper input format for the BTCUSDT 4h fine-tuned model

## How to Use

### Installation

```bash
pip install transformers torch
```

### Loading the Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("lc2004/kronos_tokenizer_base_BTCUSDT_4h_finetune")
```

### Tokenizing BTCUSDT Data

```python
# Example: Tokenize BTCUSDT candlestick data
candlestick_data = "BTCUSDT 4h: Open=45230.5, High=45600.2, Low=45100.3, Close=45450.8, Volume=9382.45"

tokens = tokenizer.encode(candlestick_data, return_tensors="pt")
print(tokens)

# Decode tokens back to readable format
decoded = tokenizer.decode(tokens[0])
print(decoded)
```

### Integration with Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lc2004/kronos_tokenizer_base_BTCUSDT_4h_finetune")
model = AutoModelForCausalLM.from_pretrained("lc2004/kronos_base_model_BTCUSDT_4h_finetune")

# Prepare data
historical_data = "OHLCV data here..."
tokens = tokenizer.encode(historical_data, return_tensors="pt")

# Get predictions
outputs = model.generate(tokens, max_length=50)
predictions = tokenizer.decode(outputs[0])
```

## Technical Specifications

- **Compatible with**: BTCUSDT 4-Hour Fine-tuned Model
- **Data Format**: Open, High, Low, Close, Volume (OHLCV)
- **Time Granularity**: 4-hour candlesticks
- **Supported Operations**: Encoding, decoding, tokenization
- **Framework**: PyTorch / TensorFlow compatible
- **Prediction Window**: Up to 48 periods (192 hours / 8 days)

## Training Data

- **Dataset**: BTCUSDT 4-hour historical candles
- **Source**: Cryptocurrency exchange data
- **Time Coverage**: Historical trading data up to October 2025
- **Data Points**: Thousands of 4-hour candles
- **Use Case**: Medium-term price forecasting

## Limitations

- **Specialized for BTCUSDT**: Not recommended for other cryptocurrency pairs or timeframes
- **4-Hour Granularity**: Designed specifically for 4-hour candlestick data
- **Numeric Focus**: Optimized for OHLCV data format
- **Normalization**: Assumes price ranges similar to historical BTCUSDT data
- **Medium-term Optimized**: Best for 3-7 day forecasting horizons

## Usage Notes

⚠️ **Important**:
- This tokenizer should be used **exclusively with the BTCUSDT 4h fine-tuned model**
- Do not use this tokenizer with other models or datasets
- Ensure your input data follows the OHLCV format
- Maintain consistent data normalization across datasets
- Use with 4-hour candlestick data only for optimal results

## Related Models

- **Fine-tuned Model**: [BTCUSDT 4h Fine-tuned Model](https://huggingface.co/lc2004/kronos_base_model_BTCUSDT_4h_finetune)
- **Base Model**: [Kronos](https://huggingface.co/antonop/Kronos-1B-MSN)
- **1h Counterpart**: [BTCUSDT 1h Tokenizer](https://huggingface.co/lc2004/kronos_tokenizer_base_BTCUSDT_1h_finetune)

## License

This tokenizer is released under the **MIT License**.

## Citation

If you use this tokenizer, please cite:

```bibtex
@misc{btcusdt_4h_tokenizer_2025,
  title={BTCUSDT 4-Hour Tokenizer},
  author={Liucong},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/lc2004/kronos_tokenizer_base_BTCUSDT_4h_finetune}}
}
```

## Acknowledgments

- Base framework: [Hugging Face Transformers](https://huggingface.co/transformers/)
- Compatible with: [BTCUSDT 4h Fine-tuned Model](https://huggingface.co/lc2004/kronos_base_model_BTCUSDT_4h_finetune)
- Original Kronos Framework: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)

## Contact & Support

For questions:
- GitHub: [Kronos-Btc-finetune](https://github.com/Liucong-sdu/Kronos-Btc-finetune)
- Hugging Face: [lc2004](https://huggingface.co/lc2004)

---

**Last Updated**: October 23, 2025