# 忘れさせる数字のリスト
list_ewc_learn=(0 1 2 3 4 5 6 7 8 9)
# 覚えさせる数字のリスト
list_forget=(0 1 2 3 4 5 6 7 8 9)

# テスト用のリスト
list_ewc_learn=(0)
list_forget=(0)


# 実行するべきコマンド
# VAEの学習
# CUDA_VISIBLE_DEVICES="0" python train_cvae.py --config mnist.yaml --data_path ./dataset
# FIM
# CUDA_VISIBLE_DEVICES="0" python calculate_fim.py --ckpt_folder results/yyyy_mm_dd_hhmmss


# すべての組わせをループで回す
for learn in ${list_ewc_learn[@]}; do
    echo "forget: $forget, learn: $learn"
    
    echo "start VAE training. no train data class is $learn"
    vae_output_str  = $(CUDA_VISIBLE_DEVICES="0" python train_cvae.py --remove_label $learn --config mnist.yaml --data_path ./dataset) 
    
    echo "start no SA, EWC calculation"
        #output から save dir を抜き取る
        vae_save_dir=$(echo "$vae_output_str" | grep -oP 'vae save dir:\K[^\n]*')
        echo "VAE save dir is $vae_save_dir"
        echo "start FIM calculation"
        # CUDA_VISIBLE_DEVICES="0" python calculate_fim.py --ckpt_folder results/yyyy_mm_dd_hhmmss
        echo "start EWC calculation"
        

    
    for forget in ${list_forget[@]}; do
        # SA を適応して書くモデルに足してEWCを適応
        echo "start SA, EWC calculation"
        # FIMは前回のを引き継ぐ
        # CUDA_VISIBLE_DEVICES="0" python train_forget.py --ckpt_folder results/yyyy_mm_dd_hhmmss --label_to_drop 0 --lmbda 100
        # output から sa vae のsave dir を抜き取る
        
        # SA　を適用したモデルにEWCを適用
            echo "SA VAE save dir is $sa_vae_save_dir"
            # CUDA_VISIBLE_DEVICES="0" python calculate_fim.py --ckpt_folder results/yyyy_mm_dd_hhmmss
            # CUDA_VISIBLE_DEVICES="0" python train_ewc.py --ckpt_folder results/yyyy_mm_dd_hhmmss

        # モデルの評価を行う
            # 10000枚の画像を生成
            # 分類機で精度を出す

    done
done

# todo 
# outputから変数を抜き取る（save dir を引き継ぐ）
# 書く実行関数の整理
# 出力から目的の数字だけを整理して抜き出す
