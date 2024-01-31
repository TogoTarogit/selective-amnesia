#!/bin/bash
# GPUの指定を変数として甘止める
cuda_num=0
# サンプルとして出力する画像の枚数
# n_samples=1000
n_samples=10000

# 結果保存用ディレクトリ
result_save_dir="./results/text_results"

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
echo "experiment content: それぞれのモデルについて評価をする 評価方法はエントロピー すでに生成されてる場合スキップするように変更" | tee -a $result_dir_name
# ファイルに変数の値を追記
echo "CUDA Number: $cuda_num" >> $result_dir_name
echo "Number of Samples: $n_samples" >> $result_dir_name
echo "------------------------------------------------" >> $result_dir_name


FILE_PATH="./2024_01_15_155632_mnist_forget_learn_test.txt" #ランダムに忘れる
FILE_PATH="./2024_01_17_172647_mnist_forget_learn_test.txt" #　多分雑音画像
FILE_PATH="./2024_01_09_mnist_forget_learn_test.txt" #テスト用
echo "FILE_PATH: $FILE_PATH" >> $result_dir_name

# テキストファイルからディレクトリのパスを抽出して配列に格納
mapfile -t directories < <(grep -oP './results/mnist/\d+_\d+_\d+_\d+/' "$FILE_PATH")

# デバッグ情報として取得したディレクトリを表示
# echo "取得したディレクトリ:"
# printf '%s\n' "${directories[@]}"

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
        CUDA_VISIBLE_DEVICES=$cuda_num python evaluate_with_classifier.py --sample_path $directory --label_of_dropped_class $LABEL >> $result_dir_name
    done
    echo "------------------------------------------------" >> $result_dir_name


done

# 生成した画像を分類し、精度を測定する処理を追加する必要があります
# TODO: ここに分類と精度測定のための追加のコマンドを記述する
