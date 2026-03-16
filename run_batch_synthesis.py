#!/usr/bin/env python3
"""
バッチモードでオリジナルFastSpeech2の音声を生成するスクリプト
"""
import subprocess
import os
import yaml
from datetime import datetime

# パラメータ設定
source_file = "preprocessed_data/LJSpeech/val.txt"
restore_step = 900000
preprocess_config = "config/LJSpeech/preprocess.yaml"
model_config = "config/LJSpeech/model.yaml"  # オリジナル設定
train_config = "config/LJSpeech/train.yaml"  # オリジナル設定

# 実験用の設定
num_sentences = 50  # val.txtの最初の何文を使用するか

# 実験用のベースディレクトリを作成（タイムスタンプ付き）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"output/result/LJSpeech_original_experiment_{timestamp}"
os.makedirs(experiment_dir, exist_ok=True)

# val.txtの最初のN文を抽出して一時ファイルを作成
temp_val_file = os.path.join(experiment_dir, "val_subset.txt")
with open(source_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[:num_sentences]

with open(temp_val_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("=" * 60)
print(f"実験ディレクトリ: {experiment_dir}")
print(f"処理する文章数: {num_sentences}")
print("オリジナルFastSpeech2で音声を生成します")
print("=" * 60)

# 一時的なtrain_config.yamlを作成（result_pathを変更）
temp_train_config = os.path.join(experiment_dir, "train_config_original.yaml")
with open(train_config, 'r') as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

# result_pathを変更
train_cfg['path']['result_path'] = experiment_dir

with open(temp_train_config, 'w') as f:
    yaml.dump(train_cfg, f)

# synthesize.pyを実行
cmd = [
    "python", "synthesize.py",
    "--mode", "batch",
    "--source", temp_val_file,
    "--restore_step", str(restore_step),
    "-p", preprocess_config,
    "-m", model_config,
    "-t", temp_train_config,
    # use_dfなし
    "--pitch_control", "1.0",
    "--energy_control", "1.0",
    "--duration_control", "1.0"
]

try:
    print(f"実行中...")
    result = subprocess.run(cmd, check=True, capture_output=False)
    print(f"✓ 生成が完了しました")
    print(f"  保存先: {experiment_dir}")
    
    # 一時的なtrain_configファイルを削除
    os.remove(temp_train_config)

except subprocess.CalledProcessError as e:
    print(f"✗ エラーが発生しました")
    print(f"  エラー内容: {e}")

print("\n" + "=" * 60)
print("音声生成が完了しました！")
print(f"結果の保存先: {experiment_dir}")
print(f"  - {num_sentences}個の音声ファイル (.wav) とスペクトログラム (.png) が保存されています")
print("=" * 60)