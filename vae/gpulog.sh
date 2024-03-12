#!/bin/bash
# usege :
# nohup bash gpulog.sh &> /dev/null &

# GPU監視スクリプト（CSV形式、複数GPU対応）

# ログファイルのパスを設定
LOG_FILE="./results/gpulog/gpu_memory_usage_log.csv"
GRAPH_FILE="./gpu_memory_usage_graph.png"

# 監視間隔（秒）
INTERVAL=60

# グラフ更新間隔（秒）
GRAPH_UPDATE_INTERVAL=900

# 実行期間（秒）
# 例えば、1時間実行したい場合は3600秒を設定
RUN_TIME=$((3600*24*7))

# 開始時刻を取得（エポック秒）
START_TIME=$(date +%s)

# 終了時刻を計算
END_TIME=$((START_TIME + RUN_TIME))

# GPUの数を取得
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

# ログファイルが存在する場合、初期化。存在しない場合は作成
if [ -f "$LOG_FILE" ]; then
    > $LOG_FILE  # ファイルが存在する場合、内容を空にする
else
    touch "$LOG_FILE"  # ファイルが存在しない場合、新しく作成
fi

# CSVヘッダーを設定
echo -n "timestamp," > $LOG_FILE
for ((i=0; i<NUM_GPUS; i++))
do
    echo -n "GPU${i}_util,GPU${i}_mem_util" >> $LOG_FILE
    if [ $i -lt $((NUM_GPUS-1)) ]; then
        echo -n "," >> $LOG_FILE
    fi
done
echo "" >> $LOG_FILE

# ログ記録の無限ループ
while true
do
    # 現在時刻が終了時刻を超えたらループを終了
    CURRENT_TIME=$(date +%s)
    if [ $CURRENT_TIME -ge $END_TIME ]; then
        echo "指定した時間に達したため、スクリプトを終了します。"
        break
    fi

    # 現在の日付と時刻を取得
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # ログデータを準備
    LOG_DATA="$TIMESTAMP"

   # 各GPUの使用率とメモリ使用率を一時ファイルに保存
    TEMP_FILE=$(mktemp)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$TEMP_FILE"

    # デバッグ: 取得したデータを出力
    # echo "取得したGPU情報:"
    # cat "$TEMP_FILE"

    # 一時ファイルを読み込んでログデータにGPU使用率およびメモリ使用率を追加
    while read -r line
    do
        IFS=',' read -r gpu_index gpu_util gpu_mem_used gpu_mem_total <<< "$line"
        mem_util=$(echo "scale=0; ($gpu_mem_used/$gpu_mem_total)*100/1" | bc) # 小数を削除して整数に

        # デバッグ: GPU利用率と計算されたメモリ使用率を出力
        # echo "GPU${gpu_index} - メモリ使用量: ${gpu_mem_used} , : メモリ最大容量${gpu_mem_total}"
        # echo "GPU${gpu_index} - GPU利用率: ${gpu_util}% , メモリ使用率: ${mem_util}%"

        LOG_DATA="$LOG_DATA,$gpu_util,$mem_util"
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
