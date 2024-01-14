import matplotlib.pyplot as plt
import pandas as pd
import sys

def generate_gpu_usage_graph(csv_file_path, output_graph_path):
    # CSVファイルを読み込む
    data = pd.read_csv(csv_file_path)

    # タイムスタンプをインデックスに設定
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # 各GPUの使用率をプロット
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)

    # グラフの設定
    plt.xlabel('Time')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # グラフをファイルに保存
    plt.savefig(output_graph_path)
    plt.close()

if __name__ == "__main__":
    # コマンドライン引数からファイルパスを取得
    csv_file_path = sys.argv[1]
    output_graph_path = sys.argv[2]

    # グラフ生成関数を呼び出し
    generate_gpu_usage_graph(csv_file_path, output_graph_path)
    
# python visualizegpu.py ./gpu_usage_log.csv ./output_graph.png

