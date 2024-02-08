

# 文章を読み取り、labelとprobの辞書を作成する関数
def create_label_prob_dict(file_path):
    label_prob_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        label = None
        prob = None
        for line in lines:
            if line.startswith("label:"):
                label = int(line.split(":")[1].strip())
            elif line.startswith("Average prob of forgotten class:"):
                prob = float(line.split(":")[1].strip())
                label_prob_dict[label] = prob
    return label_prob_dict

# ファイルパス
file_path = "./input.txt"
Df = 8
Dnew = 3
print("Df,Dnew",Df,Dnew)
# labelとprobの辞書を作成
label_prob_dict = create_label_prob_dict(file_path)

# 辞書の出力
# print(label_prob_dict)

def calculate_accuracy(Df, Dnew, label_prob_dict):
    prob_Df = label_prob_dict.get(Df, 0)  # Dfの確率を取得する。存在しない場合は0を返す。
    prob_Dnew = label_prob_dict.get(Dnew, 0)  # Dnewの確率を取得する。存在しない場合は0を返す。
    
    
    # 精度を計算する
    print("prob_Df:", prob_Df)
    total_prob = 0
    count = 0
    for label, prob in label_prob_dict.items():
        total_prob += prob
        count += 1
    average_D = total_prob / count
    prob_Dr = (total_prob-prob_Df-prob_Dnew) /8
    print("Dr:",prob_Dr)
    print("prob_Dnew:", prob_Dnew)
    
    return 1


calculate_accuracy(Df, Dnew, label_prob_dict)