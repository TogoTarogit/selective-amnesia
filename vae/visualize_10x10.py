import matplotlib.pyplot as plt
import numpy as np
import datetime
# ファイルを読み込んで分析する
# file_path = "./2024_01_12_212528_mnist_forget_learn_test.txt"
# file_path = "./2024_01_09_mnist_forget_learn_test.txt" 
# file_path = "./2024_01_15_155632_mnist_forget_learn_test.txt"# random image forget
# file_path = "./2024_01_14_132849_mnist_forget_learn_test.txt" # white noise image forget  include bug
# file_path = "./2024_01_17_172647_mnist_forget_learn_test.txt"
# file_path = "./results/text_results/2024_02_01_121854_fix_mnist_forget_learn_test_mnist_noise.txt"
# file_path = "./mnist_noise_fix.txt"
file_path = "./fashion_noise_fix.txt"

# file_path = ""


# 結果を保持するための辞書を準備
results = {
    "sa_ewc": {},
    "nosa_ewc": {},
    "sa_finetuning": {},
    "finetuning": {}
}

# 現在の日時を取得（年_月_日_時_分の形式）
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H")
contents = "\n"+"When given a whitenouse image during the forgetting process"
file_name = "\n" +file_path
subtitle = current_time + contents + file_name
# ファイルを読み込む
with open(file_path, 'r') as file:
    lines = file.readlines()
    # print(lines)
    # 現在の実験タイプとラベルを追跡
    current_experiment = None
    current_labels = None

    # 実験タイプをマッピングする辞書
    EXPERIMENT_TYPES = {
        "sa,ewc": "sa_ewc",
        "nosa,ewc": "nosa_ewc",
        "sa,finetuning": "sa_finetuning",
        "finetuning": "finetuning"
    }

    for line in lines:
        line = line.strip()
        # 実験タイプの識別
        for key in EXPERIMENT_TYPES:
            if line.startswith(key):
                current_experiment = EXPERIMENT_TYPES[key]
                print(current_experiment)
                break

        if line.startswith("forget:") and "learn:" in line:
            # ラベルの変更
            forget_label, learn_label = [int(part.split(":")[1].strip()) for part in line.split(",")]
            current_labels = (forget_label, learn_label)
            print(current_labels)

        elif line.startswith("Average prob of forgotten class:"):
            # 確率値を抽出
            prob = float(line.split(":")[1].strip())
            if current_experiment and current_labels:
                if current_labels not in results[current_experiment]:
                    results[current_experiment][current_labels] = prob

        print("--------------------------------------------------")
# 10x10の行列を作成
matrix_sa_ewc = [[0 for _ in range(10)] for _ in range(10)]
matrix_nosa_ewc = [[0 for _ in range(10)] for _ in range(10)]
matrix_sa_finetuning = [[0 for _ in range(10)] for _ in range(10)]
matrix_finetuning = [[0 for _ in range(10)] for _ in range(10)]

for forget in range(10):
    for learn in range(10):
        matrix_sa_ewc[forget][learn] = results["sa_ewc"].get((forget, learn), 1)
        matrix_nosa_ewc[forget][learn] = results["nosa_ewc"].get((forget, learn), 1)
        matrix_sa_finetuning[forget][learn] = results["sa_finetuning"].get((forget, learn), 1)
        matrix_finetuning[forget][learn] = results["finetuning"].get((forget, learn), 1)


# print(matrix_sa_ewc)
# print("--------------------------------")
# print(matrix_nosa_ewc)

# SA EWCの行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_sa_ewc, cmap='hot', interpolation='nearest')
plt.title('SA EWC' + subtitle)
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])

# 行列の値を表示
for i in range(10):
    for j in range(10):
        color = 'black' if matrix_sa_ewc[i][j] > 0.3 else 'white'
        plt.text(j, i, round(matrix_sa_ewc[i][j], 2), ha='center', va='center', color=color)

plt.savefig(f'./{current_time}_matrix_sa_ewc_values.png')
plt.close()

# NoSA EWCの行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_nosa_ewc, cmap='hot', interpolation='nearest')
plt.title('NoSA EWC' + subtitle)
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
for i in range(10):
    for j in range(10):
        color = 'black' if matrix_nosa_ewc[i][j] > 0.3 else 'white'
        plt.text(j, i, round(matrix_nosa_ewc[i][j], 2), ha='center', va='center', color=color)
plt.savefig(f'./{current_time}_matrix_nosa_ewc_values.png')
plt.close()


# SA EWCとNoSA EWCの行列の差分を計算
# 行列の差分を計算
matrix_diff = np.array(matrix_sa_ewc) - np.array(matrix_nosa_ewc)
max_abs_value = np.max(np.abs(matrix_diff))
# 図の最大値を調整する変数
max_image_ratio = 1.2

# EWC　の差分行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_diff, cmap='RdBu', interpolation='nearest',vmin=-max_abs_value*max_image_ratio, vmax=max_abs_value*max_image_ratio)
plt.title('Difference between SA EWC and NoSA EWC' +subtitle)
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
# 行列の値を表示
for i in range(10):
    for j in range(10):
        color = 'black'
        plt.text(j, i, round(matrix_diff[i][j], 2), ha='center', va='center', color=color)

plt.savefig(f'./{current_time}_matrix_diff_ewc_values.png')
plt.close()


# 差分行列をプロット（論文用の出力）
plt.rcParams.update({'font.size': 12})  # ここでフォントサイズを調整
max_abs_value_conf = np.max(np.abs(matrix_diff))
plt.figure(figsize=(10, 10))
img = plt.imshow(matrix_diff, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value_conf*max_image_ratio, vmax=max_abs_value_conf *max_image_ratio)
# plt.colorbar()
plt.xticks(range(10), [f'Dnew {i}' for i in range(10)])
plt.yticks(range(10), [f'Df {i}' for i in range(10)])
# 対角要素に斜線を描画
for i in range(10):
    plt.gca().add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, color="black"))
    # plt.plot([i-0.5, i+0.5], [i-0.5, i+0.5], color="k", linestyle="-")
    # plt.plot([i-0.5, i+0.5], [i+0.5, i-0.5], color="k", linestyle="-")
# 非対角要素に数値を表示
for i in range(10):
    for j in range(10):
        if i != j:  # 対角要素以外
            color = 'black'
            plt.text(j, i, round(matrix_diff[i][j], 2), ha='center', va='center', color=color)
plt.tight_layout()  # 図の余白を調整
plt.savefig(f'./{current_time}_conf_matrix_diff_ewc_values.png')
plt.close()

# カラーバー専用の図と軸を作成
# plt.tight_layout()
fig, ax = plt.subplots(figsize=(1,10),constrained_layout=True)
# fig.subplots_adjust(rigat=0.1)
# ax = fig.add_axes([0.85, 0.1, 0.03, 0.8]) 
# カラーバーを描画
cbar = plt.colorbar(img, cax=ax, orientation='vertical')
# カラーバーの目盛りラベルのスタイルを調整
cbar.ax.tick_params(pad=5)  # ラベルのフォントサイズを調整

# カラーバーを画像として保存
plt.savefig(f'./{current_time}_colorbar.png')
plt.close()

# finetuningの行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_finetuning, cmap='hot', interpolation='nearest')
plt.title('Finetuning' + subtitle)
plt.colorbar()
plt.xticks(range(10), [f'Dnew {i}' for i in range(10)])
plt.yticks(range(10), [f'Df {i}' for i in range(10)])
for i in range(10):
    for j in range(10):
        color = 'black' if matrix_finetuning[i][j] > 0.3 else 'white'
        plt.text(j, i, round(matrix_finetuning[i][j], 2), ha='center', va='center', color=color)

plt.savefig(f'./{current_time}_matrix_finetuning_values.png')
plt.close()

# SA Finetuningの行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_sa_finetuning, cmap='hot', interpolation='nearest')
plt.title('SA Finetuning'+subtitle)
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
for i in range(10):
    for j in range(10):
        color = 'black' if matrix_sa_finetuning[i][j] > 0.3 else 'white'
        plt.text(j, i, round(matrix_sa_finetuning[i][j], 2), ha='center', va='center', color=color)
plt.savefig(f'./{current_time}_matrix_sa_finetuning_values.png')
plt.close()

# Finetuning と SA Finetuning の行列の差分を計算
matrix_diff_finetuning = np.array(matrix_sa_finetuning) - np.array(matrix_finetuning)
max_abs_value_finetuning = np.max(np.abs(matrix_diff_finetuning))

# 差分行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_diff_finetuning, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value_finetuning*max_image_ratio, vmax=max_abs_value_finetuning*max_image_ratio)
plt.title('Difference between SA Finetuning and Finetuning' + subtitle)
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
for i in range(10):
    for j in range(10):
        color = 'black'
        plt.text(j, i, round(matrix_diff_finetuning[i][j], 2), ha='center', va='center', color=color)

plt.savefig(f'./{current_time}_matrix_diff_finetuning_values.png')
plt.close()

# 差分行列をプロット　論文用
plt.rcParams.update({'font.size': 12})  # ここでフォントサイズを調整
plt.figure(figsize=(10, 10))
plt.imshow(matrix_diff_finetuning, cmap='RdBu', interpolation='nearest', vmin=-max_abs_value_conf *max_image_ratio, vmax=max_abs_value_conf *max_image_ratio)
# plt.title('Difference between SA Finetuning and Finetuning' + subtitle)
# plt.colorbar()
plt.xticks(range(10), [f'Dnew {i}' for i in range(10)])
plt.yticks(range(10), [f'Df {i}' for i in range(10)])
# 対角要素の表現
for i in range(10):
    plt.gca().add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, color="black"))
    # plt.plot([i-0.5, i+0.5], [i-0.5, i+0.5], color="k", linestyle="-")
    # plt.plot([i-0.5, i+0.5], [i+0.5, i-0.5], color="k", linestyle="-")

for i in range(10):
    for j in range(10):
        if i != j:  # 対角要素以外
            color = 'black'
            plt.text(j, i, round(matrix_diff_finetuning[i][j], 2), ha='center', va='center', color=color)
plt.tight_layout()  # 図の余白を調整
plt.savefig(f'./{current_time}_conf_matrix_diff_finetuning_values.png')
plt.close()