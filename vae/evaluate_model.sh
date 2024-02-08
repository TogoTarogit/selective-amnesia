#!/bin/bash
# GPUの指定を変数として甘止める
cuda_num=1
# サンプルとして出力する画像の枚数
# ----------------------------------------------------
# 実験設定
# n_samples=1000
n_samples=10000

# 結果保存用ディレクトリ
result_save_dir="./results/text_results"
contents_discription=""
dataset="mnist" 
# ----------------------------------
# 結果保存用ディレクトリが存在しない場合は作成
if [ ! -d "$result_save_dir" ]; then
    mkdir -p "$result_save_dir"
fi
# 結果出保存用のファイルを作成，名前は日付
file_name=$(date "+%Y_%m_%d_%H%M%S")_mnist_forget_learn_test.txt
result_dir_name="$result_save_dir/$file_name"

# 実験の日付を表示
echo "experiment date: $(date "+%Y/%m/%d %H:%M:%S")" | tee -a $result_dir_name
# 実験の内容について記録
echo "experiment content: $contents_discription" | tee -a $result_dir_name
# ファイルに変数の値を追記
echo "CUDA Number: $cuda_num" >> $result_dir_name
echo "Number of Samples: $n_samples" >> $result_dir_name
echo "dataset is $dataset" >> $result_dir_name


# FILE_PATH="./" #テスト用
# FILE_PATH="./results/text_results/2024_02_02_192716_mnist_forget_learn_test.txt" #mnist randam
# FILE_PATH="./results/text_results/2024_02_02_192655_mnist_forget_learn_test.txt" #fashion random
# FILE_PATH="./results/text_results/2024_02_01_121854_mnist_forget_learn_test.txt" #mnist noise
# FILE_PATH="./results/text_results/2024_02_01_121854_fix_mnist_forget_learn_test_mnist_noise.txt" #mnist noise
            # /results/text_results/2024_02_01_121854_fix_mnist_forget_learn_test_mnist_noise.txt
# FILE_PATH="./fashion_noise_fix.txt" #
# FILE_PATH="./fashion_random_fix.txt" #
FILE_PATH="./mnist_random_fix.txt" #

echo "FILE_PATH: $FILE_PATH" >> $result_dir_name
# ファイルが存在するかチェック
if [ -f "$FILE_PATH" ]; then
    # ファイルの内容を出力
    cat "$FILE_PATH"
else
    echo "ファイルが存在しません: $FILE_PATH"
fi
# cat $FILE_PATH >> $result_dir_name
# mapfile -t directories < <(grep -oP './results/mnist/\d+_\d+_\d+_\d+/' "$FILE_PATH")

# テキストファイルからディレクトリのパスを抽出して配列に格納
# mapfile -t directories < <(grep -oP './results/mnist/(\d+_\d+_\d+_\d+)/' "$FILE_PATH")
# mapfile -t directories < <(grep -oP './results/mnist/(\d+_)+/' "$FILE_PATH")
# datasetに応じて適切な値を設定
if [ "$dataset" = "mnist" ]; then
    dataset_dir="mnist"
elif [ "$dataset" = "fashion" ]; then
    dataset_dir="fashionmnist"
else
    echo "不明なdataset: $dataset"
    exit 1
fi
mapfile -t directories < <(grep -oP "./results/${dataset_dir}/(\d+_)+\d+" "$FILE_PATH")
# デバッグ情報として取得したディレクトリを表示
printf '%s\n' "${directories[@]}"
echo "------------------------------------------------" >> $result_dir_name

# echo "取得したディレクトリ:"

# ディレクトリの数だけループ
for directory in "${directories[@]}"
do 
    # テキストファイルからディレクトリが書かれている行を抽出
    # その行の２行前のデータを保存
    experiment_type=$(grep -B 2 "$directory" "$FILE_PATH" | head -n 1)
    # その行の１行前のデータを保存
    forget_learn=$(grep -B 1 "$directory" "$FILE_PATH" | head -n 1)
    echo "experiment_type: $experiment_type" >> $result_dir_name
    echo "forget_learn: $forget_learn" >> $result_dir_name
    echo "model path : $directory" | tee -a $result_dir_name
    
    # 0から9までのラベルに対してループ
    for LABEL in {0..9}
    do
        echo "label: $LABEL" >> $result_dir_name
        # サンプル生成の出力をファイルに追記
        # echo "Generating samples for label $LABEL..." >> $result_dir_name
        CUDA_VISIBLE_DEVICES=$cuda_num python generate_samples.py --ckpt_folder $directory --label_to_generate $LABEL --n_samples $n_samples >> $result_dir_name 
        
        # サンプル評価の出力をファイルに追記
        # echo "Evaluating samples for label $LABEL..." >> $result_dir_name
        # CUDA_VISIBLE_DEVICES=$cuda_num python evaluate_with_classifier.py --sample_path $directory --label_of_dropped_class $LABEL >> $result_dir_name
        CUDA_VISIBLE_DEVICES=$cuda_num python evaluate_with_classifier.py --sample_path $directory --label_of_dropped_class $LABEL --dataset $dataset >> $result_dir_name

    done
    echo "------------------------------------------------" >> $result_dir_name


done

# 生成した画像を分類し、精度を測定する処理を追加する必要があります
# TODO: ここに分類と精度測定のための追加のコマンドを記述する
