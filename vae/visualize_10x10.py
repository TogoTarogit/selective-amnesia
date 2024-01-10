import matplotlib.pyplot as plt
import numpy as np
# ファイルを読み込んで分析する
file_path = "./2024_01_09_mnist_forget_learn_test.txt"

# 結果を保持するための辞書を準備
results = {
    "sa_ewc": {},
    "nosa_ewc": {}
}

# ファイルを読み込む
with open(file_path, 'r') as file:
    lines = file.readlines()
    # print(lines)
    # 現在の実験タイプとラベルを追跡
    current_experiment = None
    current_labels = None

    for line in lines:
        line = line.strip()
        if line.startswith("sa,ewc") or line.startswith("nosa,ewc"):
            # 実験タイプの変更
            current_experiment = "sa_ewc" if line.startswith("sa,ewc") else "nosa_ewc"
            print(current_experiment)
        elif line.startswith("forget:") and "learn:" in line:
            # ラベルの変更
            parts = line.split(",")
            forget_label = int(parts[0].split(":")[1].strip())
            learn_label = int(parts[1].split(":")[1].strip())
            current_labels = (forget_label, learn_label)
            print(current_labels)
        elif line.startswith("Average prob of forgotten class:"):
            # 確率値を抽出
            prob = float(line.split(":")[1].strip())
            if current_experiment and current_labels:
                if current_labels not in results[current_experiment]:
                    results[current_experiment][current_labels] = prob
        # print(results)
        print("--------------------------------------------------")            

# 10x10の行列を作成
matrix_sa_ewc = [[0 for _ in range(10)] for _ in range(10)]
matrix_nosa_ewc = [[0 for _ in range(10)] for _ in range(10)]

for forget in range(10):
    for learn in range(10):
        matrix_sa_ewc[forget][learn] = results["sa_ewc"].get((forget, learn), 1)
        matrix_nosa_ewc[forget][learn] = results["nosa_ewc"].get((forget, learn), 1)

# print(matrix_sa_ewc)
# print("--------------------------------")
# print(matrix_nosa_ewc)

# SA EWCの行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_sa_ewc, cmap='hot', interpolation='nearest')
plt.title('SA EWC')
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])

# 行列の値を表示
for i in range(10):
    for j in range(10):
        color = 'black' if matrix_sa_ewc[i][j] > 0.3 else 'white'
        plt.text(j, i, round(matrix_sa_ewc[i][j], 2), ha='center', va='center', color=color)

plt.savefig('./matrix_sa_ewc_values.png')
plt.close()

# NoSA EWCの行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_nosa_ewc, cmap='hot', interpolation='nearest')
plt.title('NoSA EWC')
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])
for i in range(10):
    for j in range(10):
        color = 'black' if matrix_nosa_ewc[i][j] > 0.3 else 'white'
        plt.text(j, i, round(matrix_nosa_ewc[i][j], 2), ha='center', va='center', color=color)
plt.savefig('./matrix_nosa_ewc_values.png')
plt.close()


# SA EWCとNoSA EWCの行列の差分を計算
# 行列の差分を計算
matrix_diff = np.array(matrix_sa_ewc) - np.array(matrix_nosa_ewc)
max_abs_value = np.max(np.abs(matrix_diff))
# 図の最大値を調整する変数
max_image_ratio = 1.0 + 0.1
# 差分行列をプロット
plt.figure(figsize=(10, 10))
plt.imshow(matrix_diff, cmap='RdBu', interpolation='nearest',vmin=-max_abs_value*max_image_ratio, vmax=max_abs_value*max_image_ratio)
plt.title('Difference between SA EWC and NoSA EWC')
plt.colorbar()
plt.xticks(range(10), [f'learn_{i}' for i in range(10)])
plt.yticks(range(10), [f'forget_{i}' for i in range(10)])

# 行列の値を表示
for i in range(10):
    for j in range(10):
        color = 'black'
        plt.text(j, i, round(matrix_diff[i][j], 2), ha='center', va='center', color=color)

plt.savefig('./matrix_diff_values.png')
plt.close()