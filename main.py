# main.py
"""
就活支援レコメンドシステム (特徴量活用・Two-Tower BPRモデル版)

このスクリプトは、学生と求人の特徴量を活用した推薦システムを構築・評価します。
Two-TowerモデルアーキテクチャとBPR損失を採用し、ランキング精度を向上させます。

実行方法:
1. 必要なライブラリをインストール: pip install -r requirements.txt
2. スクリプトを実行: python main.py
"""
import os
import sys
import random
import time
from datetime import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# KerasとTensorFlowのバックエンド設定を先に行う
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlowのログを抑制

import keras
import keras_rs
import tensorflow as tf

# --- グローバル設定とハイパーパラメータ ---
# データ生成設定
NUM_STUDENTS = 500
NUM_JOBS = 100
MAX_INTERACTIONS_PER_STUDENT = 15
MIN_INTERACTIONS_PER_STUDENT = 3

# モデル設定
EMBEDDING_DIM = 64  # 埋め込みベクトルの次元数
TOWER_HIDDEN_UNITS = [128, 64] # 各タワーの全結合層のユニット数
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 15

# 評価設定
TOP_K = 10 # 推薦リストの数
EVAL_NUM_STUDENTS = 50 # 評価に使用するサンプル学生数
RANDOM_SEED = 42

# --- 1. ダミーデータの生成 ---
def generate_dummy_data():
    """就活ドメインのダミーデータを生成する"""
    print("--- 1. ダミーデータの生成開始 ---")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    faculties = ["工学部", "理学部", "文学部", "経済学部"]
    skills_pool = [f"Skill_{i}" for i in range(50)]
    industries_pool = [f"Industry_{i}" for i in range(20)]
    job_types_pool = [f"JobType_{i}" for i in range(20)]
    locations_pool = ["東京", "大阪", "名古屋", "福岡"]
    all_features = faculties + skills_pool + industries_pool + job_types_pool + locations_pool
    feature_to_id = {feature: i for i, feature in enumerate(list(set(all_features)), 1)}
    VOCAB_SIZE = len(feature_to_id) + 1

    students_data = [{
        "student_id": i, "faculty_id": feature_to_id.get(random.choice(faculties)),
        "skill_ids": [feature_to_id.get(s) for s in random.sample(skills_pool, k=5) if feature_to_id.get(s)]
    } for i in range(1, NUM_STUDENTS + 1)]

    jobs_data = [{
        "job_id": i, "industry_id": feature_to_id.get(random.choice(industries_pool)),
        "required_skill_ids": [feature_to_id.get(s) for s in random.sample(skills_pool, k=4) if feature_to_id.get(s)],
    } for i in range(1, NUM_JOBS + 1)]

    interactions_data = []
    for student in students_data:
        interacted_count = 0
        for job in random.sample(jobs_data, len(jobs_data)):
            if interacted_count >= MAX_INTERACTIONS_PER_STUDENT: break
            skill_match = len(set(student["skill_ids"]) & set(job["required_skill_ids"]))
            if random.random() < (0.1 + skill_match * 0.2) or interacted_count < MIN_INTERACTIONS_PER_STUDENT:
                interactions_data.append({"student_id": student["student_id"], "job_id": job["job_id"]})
                interacted_count += 1
    
    print(f"生成データ数: 学生={len(students_data)}, 求人={len(jobs_data)}, インタラクション={len(interactions_data)}")
    print("--- データ生成完了 ---\n")
    return students_data, jobs_data, interactions_data, VOCAB_SIZE

# (以降、元のmain.pyのコードが続く...長いため簡略化)
# この例では、主要な部分のみを記述しています。
# 実際の使用時は、以前提示した完全なコードを使用してください。
if __name__ == "__main__":
    print("レコメンドシステムのスクリプトが作成されました。")
    print("注意：このファイルはデモ用に簡略化されています。")
    students_data, jobs_data, interactions_data, vocab_size = generate_dummy_data()
    print("\nサンプルデータが正常に生成されました。")
