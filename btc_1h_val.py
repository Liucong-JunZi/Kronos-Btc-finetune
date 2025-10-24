import time
import btc_1h_prediction as pred
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import threading

# 全局变量
app = Dash(__name__)
global_df = None
global_predicted = None
tokenizer = None
model = None
update_interval = 3600  # 默认1小时（秒）- 这里改成3600秒，不是60

def vision_start(df, predicted_data):
    """初始化可视化"""
    global global_df, global_predicted
    global_df = df.copy()
    global_predicted = predicted_data.copy()
    print("✅ 可视化初始化完成")
    print(f"   历史数据: {len(global_df)} 条")
    print(f"   预测数据: {len(global_predicted)} 条")

def vision_update(df, predicted_data):
    """更新可视化数据"""
    global global_df, global_predicted
    global_df = df.copy()
    global_predicted = predicted_data.copy()
    print(f"🔄 数据已更新 - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   最新价格: ${df['close'].iloc[-1]:.2f}")
    print(f"   预测价格: ${predicted_data['close'].iloc[-1]:.2f}")

def create_figure(df, predicted_data):
    """创建K线图"""
    if df is None or predicted_data is None:
        return go.Figure()
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('BTC/USDT 价格预测 (1小时)', '成交量'),
        row_heights=[0.7, 0.3]
    )
    
    # 获取最近100条历史数据用于显示
    hist_data = df.tail(100).copy()
    
    # === 1. 添加历史K线（实线，鲜艳颜色）===
    fig.add_trace(
        go.Candlestick(
            x=hist_data['timestamps'],
            open=hist_data['open'],
            high=hist_data['high'],
            low=hist_data['low'],
            close=hist_data['close'],
            name='📊 历史数据',
            increasing_line_color='#00FF00',  # 亮绿色
            decreasing_line_color='#FF0000',  # 亮红色
            increasing_fillcolor='#00FF00',
            decreasing_fillcolor='#FF0000',
            line=dict(width=1.5)
        ),
        row=1, col=1
    )
    
    # === 2. 添加预测K线（虚线样式，半透明）===
    if predicted_data is not None and not predicted_data.empty:
        fig.add_trace(
            go.Candlestick(
                x=predicted_data.index,
                open=predicted_data['open'],
                high=predicted_data['high'],
                low=predicted_data['low'],
                close=predicted_data['close'],
                name='🔮 AI预测',
                increasing_line_color='rgba(0, 255, 255, 0.7)',  # 青色半透明
                decreasing_line_color='rgba(255, 165, 0, 0.7)',  # 橙色半透明
                increasing_fillcolor='rgba(0, 255, 255, 0.2)',
                decreasing_fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(width=1)
            ),
            row=1, col=1
        )
        
        # === 3. 添加预测区域的分隔线（明显标记）===
        last_hist_time = hist_data['timestamps'].iloc[-1]
        last_hist_price = hist_data['close'].iloc[-1]
        first_pred_price = predicted_data['close'].iloc[0]
        
        fig.add_shape(
            type="line",
            x0=last_hist_time,
            y0=hist_data['low'].min() * 0.99,
            x1=last_hist_time,
            y1=hist_data['high'].max() * 1.01,
            line=dict(color="yellow", width=3, dash="dash"),
            row=1, col=1
        )
        
        # 添加标注
        fig.add_annotation(
            x=last_hist_time,
            y=hist_data['high'].max() * 1.01,
            text="⚡ 预测起点",
            showarrow=False,
            yshift=10,
            font=dict(size=12, color="yellow"),
            bgcolor="rgba(0,0,0,0.7)",
            row=1, col=1
        )
        
        # === 4. 添加连接线（从历史到预测）===
        fig.add_trace(
            go.Scatter(
                x=[last_hist_time, predicted_data.index[0]],
                y=[last_hist_price, first_pred_price],
                mode='lines+markers',
                line=dict(color='yellow', width=2, dash='dot'),
                marker=dict(size=8, color='yellow'),
                name='🔗 连接线',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # === 5. 添加成交量柱状图（只显示历史数据的成交量）===
    colors = ['#00FF00' if row['close'] >= row['open'] else '#FF0000' 
              for _, row in hist_data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=hist_data['timestamps'],
            y=hist_data['volume'],
            name='成交量',
            marker_color=colors,
            marker_line_width=0,
            showlegend=False,
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # === 6. 更新布局 ===
    current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.update_layout(
        title={
            'text': f'🚀 BTC/USDT 1小时K线预测 | 更新: {current_time}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#00FF00', 'family': 'Arial Black'}
        },
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1a1a1a',
        xaxis_rangeslider_visible=False,
        height=900,
        hovermode='x unified',
        font=dict(color='#E0E0E0', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.7)',
            font=dict(size=14, color='white')
        )
    )
    
    # 更新 X 轴
    fig.update_xaxes(
        gridcolor='#333333',
        showgrid=True,
        gridwidth=0.5,
        row=1, col=1
    )
    fig.update_xaxes(
        gridcolor='#333333',
        showgrid=True,
        title_text='时间',
        title_font=dict(size=14, color='#00FF00'),
        row=2, col=1
    )
    
    # 更新 Y 轴
    fig.update_yaxes(
        gridcolor='#333333',
        showgrid=True,
        gridwidth=0.5,
        title_text='价格 (USDT)',
        title_font=dict(size=14, color='#00FF00'),
        row=1, col=1
    )
    fig.update_yaxes(
        gridcolor='#333333',
        showgrid=True,
        title_text='成交量',
        title_font=dict(size=14, color='#00FF00'),
        row=2, col=1
    )
    
    return fig

# Dash 布局
app.layout = html.Div([
    # 标题区域
    html.Div([
        html.H1(
            '🚀 BTC/USDT 实时预测监控系统',
            style={
                'textAlign': 'center',
                'color': '#00FF00',
                'marginBottom': '5px',
                'fontWeight': 'bold',
                'textShadow': '0 0 10px #00FF00'
            }
        ),
        html.P(
            '⚡ Powered by Kronos Fine-tuned Model ⚡',
            style={
                'textAlign': 'center',
                'color': '#FFD700',
                'fontSize': '16px',
                'marginTop': '0',
                'fontWeight': 'bold'
            }
        )
    ]),
    
    # 控制面板
    html.Div([
        html.Div([
            html.Label('⏱️ 刷新间隔（分钟）：', 
                      style={'color': '#00FF00', 'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Input(
                id='interval-input',
                type='number',
                value=60,  # 默认60分钟
                min=1,
                max=1440,
                step=1,
                style={
                    'width': '100px',
                    'marginLeft': '10px',
                    'marginRight': '20px',
                    'backgroundColor': '#1E1E1E',
                    'color': '#00FF00',
                    'border': '2px solid #00FF00',
                    'borderRadius': '5px',
                    'padding': '5px',
                    'fontSize': '16px'
                }
            ),
            html.Button(
                '✅ 应用',
                id='apply-interval-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#00FF00',
                    'color': 'black',
                    'border': 'none',
                    'borderRadius': '5px',
                    'padding': '8px 20px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer'
                }
            ),
            html.Span(
                id='interval-status',
                style={'marginLeft': '20px', 'color': '#FFD700', 'fontSize': '14px'}
            )
        ], style={'textAlign': 'center', 'marginBottom': '15px'})
    ], style={
        'backgroundColor': '#1E1E1E',
        'padding': '15px',
        'borderRadius': '10px',
        'marginBottom': '15px',
        'border': '2px solid #00FF00'
    }),
    
    # 统计信息区域
    html.Div(id='stats-info', style={
        'textAlign': 'center',
        'color': '#E0E0E0',
        'marginBottom': '20px',
        'padding': '20px',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '10px',
        'boxShadow': '0 0 20px rgba(0,255,0,0.3)',
        'border': '2px solid #00FF00'
    }),
    
    # K线图
    dcc.Graph(id='live-prediction-graph', style={'height': '85vh'}),
    
    # 自动刷新组件
    dcc.Interval(
        id='interval-component',
        interval=3600*1000,  # 初始1小时（毫秒）
        n_intervals=0
    ),
    
    # 更新时间信息
    html.Div(id='last-update', style={
        'textAlign': 'center',
        'color': '#888',
        'marginTop': '15px',
        'fontSize': '14px',
        'padding': '10px',
        'backgroundColor': '#1E1E1E',
        'borderRadius': '5px'
    }),
], style={
    'backgroundColor': '#0E1117',
    'minHeight': '100vh',
    'padding': '20px'
})

# 回调：更新刷新间隔
@app.callback(
    [Output('interval-component', 'interval'),
     Output('interval-status', 'children')],
    [Input('apply-interval-btn', 'n_clicks')],
    [State('interval-input', 'value')]
)
def update_interval_callback(n_clicks, interval_minutes):  # 改个名字，避免和全局变量冲突
    """更新刷新间隔"""
    global update_interval
    
    if n_clicks == 0:
        return 3600*1000, ''
    
    # 验证输入
    if interval_minutes is None or interval_minutes <= 0:
        return 3600*1000, '❌ 请输入有效的时间（1-1440分钟）'
    
    # 转换为秒
    update_interval = int(interval_minutes * 60)
    interval_ms = update_interval * 1000
    
    status_msg = f'✅ 已设置：每 {interval_minutes} 分钟刷新一次'
    print(f"\n⚙️  刷新间隔已更新: {interval_minutes} 分钟 ({update_interval} 秒)")
    
    return interval_ms, status_msg

# 回调：更新图表
@app.callback(
    [Output('live-prediction-graph', 'figure'),
     Output('stats-info', 'children'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    """更新图表回调函数"""
    global global_df, global_predicted, update_interval
    
    if global_df is None or global_predicted is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='⏳ 等待数据加载...',
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117'
        )
        return empty_fig, "正在加载数据...", ""
    
    # 创建图表
    fig = create_figure(global_df, global_predicted)
    
    # 计算统计信息
    current_price = global_df['close'].iloc[-1]
    current_time = global_df['timestamps'].iloc[-1]
    
    # 预测的最后一个价格
    pred_last_price = global_predicted['close'].iloc[-1]
    pred_last_time = global_predicted.index[-1]
    
    # 价格变化
    price_change = pred_last_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # 预测时间范围
    pred_hours = len(global_predicted)
    
    # 构建统计信息显示
    stats = html.Div([
        html.Div([
            html.Div([
                html.Div('📊 当前价格', style={'color': '#888', 'fontSize': '14px'}),
                html.Div(f'${current_price:.2f}', style={
                    'color': '#00FF00',
                    'fontSize': '32px',
                    'fontWeight': 'bold',
                    'textShadow': '0 0 10px #00FF00'
                })
            ], style={'display': 'inline-block', 'marginRight': '60px', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div('🔮 预测价格', style={'color': '#888', 'fontSize': '14px'}),
                html.Div(f'${pred_last_price:.2f}', style={
                    'color': '#00FFFF',
                    'fontSize': '32px',
                    'fontWeight': 'bold',
                    'textShadow': '0 0 10px #00FFFF'
                })
            ], style={'display': 'inline-block', 'marginRight': '60px', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div('📈 预期变化', style={'color': '#888', 'fontSize': '14px'}),
                html.Div(
                    f'{price_change:+.2f} ({price_change_pct:+.2f}%)',
                    style={
                        'color': '#00FF00' if price_change >= 0 else '#FF0000',
                        'fontSize': '32px',
                        'fontWeight': 'bold',
                        'textShadow': f'0 0 10px {"#00FF00" if price_change >= 0 else "#FF0000"}'
                    }
                )
            ], style={'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'marginBottom': '15px'}),
        
        html.Div([
            html.Span(f'⏱️ 预测时长: {pred_hours} 小时 | ', 
                     style={'color': '#FFD700', 'marginRight': '20px', 'fontSize': '16px', 'fontWeight': 'bold'}),
            html.Span(f'📅 预测至: {pred_last_time.strftime("%Y-%m-%d %H:%M")}', 
                     style={'color': '#FFD700', 'fontSize': '16px', 'fontWeight': 'bold'})
        ])
    ])
    
    # 更新时间信息
    now = pd.Timestamp.now()
    try:
        # 打印调试信息
        print(f"DEBUG: update_interval type = {type(update_interval)}, value = {update_interval}")
        
        # 确保 update_interval 是整数
        if callable(update_interval):
            interval_seconds = 3600  # 如果是函数，使用默认值
            print("⚠️  update_interval 是函数，使用默认值 3600 秒")
        else:
            interval_seconds = int(update_interval)
        
        if interval_seconds <= 0 or interval_seconds > 86400:
            interval_seconds = 3600
        
        next_update = now + pd.Timedelta(seconds=interval_seconds)
        interval_display = interval_seconds // 60
        
        update_info = html.Div([
            html.Span('🕐 最后更新: ', style={'color': '#888'}),
            html.Span(f'{now.strftime("%Y-%m-%d %H:%M:%S")}', 
                     style={'color': '#00FF00', 'fontWeight': 'bold', 'marginRight': '30px'}),
            html.Span(f'⏰ 下次更新: ', style={'color': '#888'}),
            html.Span(f'{next_update.strftime("%Y-%m-%d %H:%M:%S")}', 
                     style={'color': '#FFD700', 'fontWeight': 'bold', 'marginRight': '20px'}),
            html.Span(f'({interval_display} 分钟后)', 
                     style={'color': '#888', 'fontSize': '12px'})
        ])
    except Exception as e:
        print(f"⚠️  时间计算错误: {e}")
        print(f"    update_interval = {update_interval}, type = {type(update_interval)}")
        update_info = html.Div([
            html.Span('🕐 最后更新: ', style={'color': '#888'}),
            html.Span(f'{now.strftime("%Y-%m-%d %H:%M:%S")}', 
                     style={'color': '#00FF00', 'fontWeight': 'bold'})
        ])
    
    return fig, stats, update_info

def prediction_loop():
    """预测循环（后台线程）"""
    global tokenizer, model, global_df, global_predicted, update_interval
    
    print("\n" + "="*70)
    print("🚀 启动 BTC 预测循环...")
    print("="*70)
    
    # 1. 加载模型
    print("\n📦 正在加载模型...")
    tokenizer, model = pred.load_finetuned_models()
    print("✅ 模型加载完成\n")
    
    # 首次预测
    print("🔮 开始首次预测...")
    df = pred.get_latest_btc_data(days=30)
    input_data = pred.preprocess_data(df, lookback_window=512)
    predicted_data = pred.predict_btc_prices(tokenizer, model, input_data, pred_len=48)
    vision_start(df, predicted_data)
    print("✅ 首次预测完成\n")
    
    # 循环更新
    update_count = 1
    while True:
        try:
            wait_minutes = update_interval // 60
            print(f"\n⏳ 等待 {wait_minutes} 分钟后进行下一次预测... (已完成 {update_count} 次)")
            time.sleep(update_interval)
            
            print("\n" + "="*70)
            print(f"🔄 第 {update_count + 1} 次预测")
            print(f"⏰ 时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            
            # 重新获取数据
            print("📥 重新获取最新30天数据...")
            df = pred.get_latest_btc_data(days=30)
            
            # 预处理
            print("⚙️  预处理数据...")
            input_data = pred.preprocess_data(df, lookback_window=512)
            
            # 重新预测
            print("🔮 通过模型重新生成预测...")
            predicted_data = pred.predict_btc_prices(tokenizer, model, input_data, pred_len=48)
            
            # 更新可视化
            vision_update(df, predicted_data)
            
            update_count += 1
            print("✅ 预测完成\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  收到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 预测出错: {e}")
            import traceback
            traceback.print_exc()
            print("\n⏳ 5分钟后重试...")
            time.sleep(300)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 BTC/USDT 实时预测监控系统")
    print("="*70)
    print("📌 模型: Kronos Fine-tuned (1h)")
    print("📌 预测长度: 48 小时")
    print("📌 默认刷新: 每 60 分钟")
    print("📌 可在网页上自定义刷新间隔")
    print("="*70 + "\n")
    
    # 启动后台预测线程
    print("🚀 启动后台预测线程...")
    prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
    prediction_thread.start()
    
    # 等待首次数据
    print("⏳ 等待首次预测完成...")
    while global_df is None:
        time.sleep(1)
    
    print("\n" + "="*70)
    print("✅ 系统启动成功！")
    print("="*70)
    print("📊 请在浏览器中打开: http://127.0.0.1:8051")
    print("⏰ 默认每60分钟自动更新")
    print("🔄 可在网页上调整刷新间隔（1-1440分钟）")
    print("🔄 每次刷新会重新获取数据并重新预测")
    print("="*70 + "\n")
    
    # 启动 Dash 服务器
    try:
        app.run(host='127.0.0.1', port=8051, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")





