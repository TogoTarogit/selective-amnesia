#!/bin/bash

# GPU監視スクリプト（CSV形式、複数GPU対応）

# ログファイルのパスを設定
LOG_FILE="./gpu_usage_log.csv"
GRAPH_FILE="./gpu_usage_graph.png"

# 監視間隔（秒）
INTERVAL=60

# グラフ更新間隔（秒）
GRAPH_UPDATE_INTERVAL=3600

# GPUの数を取得
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

# CSVヘッダーを設定
echo -n "timestamp," >> $LOG_FILE
for ((i=0; i<NUM_GPUS; i++))
do
    echo -n "GPU$i" >> $LOG_FILE
    if [ $i -lt $((NUM_GPUS-1)) ]; then
        echo -n "," >> $LOG_FILE
    fi
done
echo "" >> $LOG_FILE

# ログ記録の無限ループ
while true
do
    # 現在の日付と時刻を取得
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # ログデータを準備
    LOG_DATA="$TIMESTAMP"

    # 各GPUの使用率を一時ファイルに保存
    TEMP_FILE=$(mktemp)
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits > "$TEMP_FILE"

    # 一時ファイルを読み込んでログデータに追加
    while read -r line
    do
        GPU_USAGE=$(echo $line | cut -d, -f2 | xargs)  # xargsは余分な空白を削除するために使用
        LOG_DATA="$LOG_DATA,$GPU_USAGE"
    done < "$TEMP_FILE"
    rm "$TEMP_FILE"  # 一時ファイルを削除

    # ログデータをファイルに書き込み
    echo "$LOG_DATA" >> $LOG_FILE
    
    # 1時間ごとにグラフを更新
    current_time=$(date +%s)
    if [ $((current_time - last_graph_update)) -ge $GRAPH_UPDATE_INTERVAL ]; then
        # グラフを生成して保存
        python visualizegpu.py $LOG_FILE $GRAPH_FILE 
        last_graph_update=$current_time
    fi

    # 指定した間隔で待機
    sleep $INTERVAL
done
