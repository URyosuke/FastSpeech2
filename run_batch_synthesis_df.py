#!/usr/bin/env python3
"""
バッチモードでdf_controlを変化させて音声を生成するスクリプト
各df_control値ごとに別のサブディレクトリに保存する
"""
import subprocess
import numpy as np
import os
import yaml
from datetime import datetime

# パラメータ設定
source_file = "preprocessed_data/LJSpeech/val.txt"
restore_step = 900000
preprocess_config = "config/LJSpeech/preprocess.yaml"
model_config = "config/LJSpeech/model_df.yaml"
train_config = "config/LJSpeech/train_df_world.yaml"

# 実験用の設定
num_sentences = 50  # val.txtの最初の何文を使用するか
df_control_values = np.arange(0.1, 6.1, 0.1)  # df_controlの値

# 実験用のベースディレクトリを作成（タイムスタンプ付き）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"output/result-df-world2/LJSpeech_df_world_experiment_{timestamp}"
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
print(f"合計 {len(df_control_values)} 個のdf_control値で音声を生成します")
print(f"df_control範囲: {df_control_values[0]:.1f} ~ {df_control_values[-1]:.1f}")
print("=" * 60)

for i, df_control in enumerate(df_control_values, 1):
    print(f"\n[{i}/{len(df_control_values)}] df_control = {df_control:.1f} で実行中...")
    
    # df_control値ごとのサブディレクトリを作成
    output_dir = os.path.join(experiment_dir, f"df_{df_control:.1f}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 一時的なtrain_config.yamlを作成（result_pathを変更）
    temp_train_config = os.path.join(experiment_dir, f"train_config_df_{df_control:.1f}.yaml")
    with open(train_config, 'r') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # result_pathを変更
    train_cfg['path']['result_path'] = output_dir
    
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
        "--use_df",  # 通常モードとdfモードはここで切り替え
        "--df_control", f"{df_control:.1f}",
        "--pitch_control", "1.0",
        "--energy_control", "1.0",
        "--duration_control", "1.0"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ df_control = {df_control:.1f} の生成が完了しました")
        print(f"  保存先: {output_dir}")
        
        # 一時的なtrain_configファイルを削除
        os.remove(temp_train_config)
        
    except subprocess.CalledProcessError as e:
        print(f"✗ エラーが発生しました (df_control = {df_control:.1f})")
        print(f"  エラー内容: {e}")
        # エラーが発生しても続行
        continue

print("\n" + "=" * 60)
print("すべての音声生成が完了しました！")
print(f"結果の保存先: {experiment_dir}")
print(f"  - 各サブディレクトリ (df_0.1, df_0.2, ..., df_6.0) に")
print(f"  - {num_sentences}個の音声ファイル (.wav) とスペクトログラム (.png) が保存されています")
print("=" * 60)

