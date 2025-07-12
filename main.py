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

# --- 2. データセットの準備 ---
def prepare_datasets(students_data, jobs_data, interactions_data):
    """TensorFlow Datasetを作成し、訓練・評価の準備をする"""
    print("--- 2. データセットの準備開始 ---")
    student_features_map = {s['student_id']: s for s in students_data}
    job_features_map = {j['job_id']: j for j in jobs_data}
    
    max_skills_student = max(len(s['skill_ids']) for s in students_data) if students_data else 10
    max_skills_job = max(len(j['required_skill_ids']) for j in jobs_data) if jobs_data else 10

    def generator():
        for interaction in interactions_data:
            student_id = interaction["student_id"]
            job_id = interaction["job_id"]
            student = student_features_map.get(student_id)
            job = job_features_map.get(job_id)
            if not student or not job: continue
            
            student_skills = student['skill_ids'] + [0] * (max_skills_student - len(student['skill_ids']))
            job_skills = job['required_skill_ids'] + [0] * (max_skills_job - len(job['required_skill_ids']))
            
            yield {
                "student_id": np.int32(student_id),
                "student_faculty_id": np.int32(student.get('faculty_id', 0)),
                "student_skill_ids": np.array(student_skills, dtype=np.int32),
                "job_id": np.int32(job_id),
                "job_industry_id": np.int32(job.get('industry_id', 0)),
                "job_skill_ids": np.array(job_skills, dtype=np.int32),
            }

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            'student_id': tf.TensorSpec(shape=(), dtype=tf.int32),
            'student_faculty_id': tf.TensorSpec(shape=(), dtype=tf.int32),
            'student_skill_ids': tf.TensorSpec(shape=(max_skills_student,), dtype=tf.int32),
            'job_id': tf.TensorSpec(shape=(), dtype=tf.int32),
            'job_industry_id': tf.TensorSpec(shape=(), dtype=tf.int32),
            'job_skill_ids': tf.TensorSpec(shape=(max_skills_job,), dtype=tf.int32),
        }
    )

    train_ds = dataset.shuffle(len(interactions_data), seed=RANDOM_SEED).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    print("--- データセット準備完了 ---\n")
    return train_ds, student_features_map, job_features_map, max_skills_student, max_skills_job, interactions_data


# --- 3. モデルの定義 (Two-Tower BPR) ---
def build_tower(inputs, vocab_size, max_skills, id_count, name):
    """学生または求人のタワーを構築する"""
    id_input = inputs[f"{name}_id"]
    feature1_input = inputs[f"{name}_faculty_id" if name == "student" else f"{name}_industry_id"]
    skills_input = inputs[f"{name}_skill_ids"]

    id_emb = keras.layers.Embedding(id_count + 1, EMBEDDING_DIM)(id_input)
    feature1_emb = keras.layers.Embedding(vocab_size, EMBEDDING_DIM // 2)(feature1_input)
    skills_emb = keras.layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(skills_input)
    skills_pooled_emb = keras.layers.GlobalAveragePooling1D()(skills_emb)

    concat = keras.layers.Concatenate()([id_emb, feature1_emb, skills_pooled_emb])
    
    x = concat
    for units in TOWER_HIDDEN_UNITS:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)
    
    output_vec = keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name=f"{name}_vector")(x)
    return output_vec

class TwoTowerBPRModel(keras.Model):
    """Two-TowerモデルとBPR損失を組み合わせた学習モデル"""
    def __init__(self, vocab_size, max_skills_student, max_skills_job, jobs_data, **kwargs):
        super().__init__(**kwargs)
        self.student_inputs = {
            "student_id": keras.Input(shape=(), dtype=tf.int32, name="student_id"),
            "student_faculty_id": keras.Input(shape=(), dtype=tf.int32, name="student_faculty_id"),
            "student_skill_ids": keras.Input(shape=(max_skills_student,), dtype=tf.int32, name="student_skill_ids"),
        }
        self.job_inputs = {
             "job_id": keras.Input(shape=(), dtype=tf.int32, name="job_id"),
             "job_industry_id": keras.Input(shape=(), dtype=tf.int32, name="job_industry_id"),
             "job_skill_ids": keras.Input(shape=(max_skills_job,), dtype=tf.int32, name="job_skill_ids"),
        }
        
        student_vector = build_tower(self.student_inputs, vocab_size, max_skills_student, NUM_STUDENTS, "student")
        job_vector = build_tower(self.job_inputs, vocab_size, max_skills_job, NUM_JOBS, "job")

        self.student_embedding_model = keras.Model(inputs=self.student_inputs, outputs=student_vector)
        self.job_embedding_model = keras.Model(inputs=self.job_inputs, outputs=job_vector)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.all_job_ids = tf.constant([j["job_id"] for j in jobs_data], dtype=tf.int32)
    
    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        student_vec = self.student_embedding_model(
            {k: data[k] for k in self.student_inputs}, training=True
        )
        positive_job_vec = self.job_embedding_model(
            {k: data[k] for k in self.job_inputs}, training=True
        )

        batch_size = tf.shape(student_vec)[0]
        random_indices = tf.random.uniform(shape=[batch_size], maxval=len(self.all_job_ids), dtype=tf.int32)
        negative_job_ids = tf.gather(self.all_job_ids, random_indices)
        
        negative_job_inputs = {
            "job_id": negative_job_ids,
            "job_industry_id": tf.zeros_like(negative_job_ids),
            "job_skill_ids": tf.zeros(shape=(batch_size, self.job_embedding_model.input_shape["job_skill_ids"][1]), dtype=tf.int32)
        }
        negative_job_vec = self.job_embedding_model(negative_job_inputs, training=True)

        with tf.GradientTape() as tape:
            pos_logits = tf.reduce_sum(student_vec * positive_job_vec, axis=1)
            neg_logits = tf.reduce_sum(student_vec * negative_job_vec, axis=1)
            loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_logits - neg_logits))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

# --- 4. 学習と評価 ---
def _dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def _ndcg_at_k(r, k):
    dcg_max = _dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return _dcg_at_k(r, k) / dcg_max

def evaluate_ranking(model, retrieval_index, student_features_map, students_data, interactions_data):
    """Precision, Recall, NDCG, MRRを計算して表示する"""
    print(f"\n--- 6. ランキング評価 (サンプル学生{EVAL_NUM_STUDENTS}名) ---")
    rng = random.Random(RANDOM_SEED)
    sample_student_ids = rng.sample([s["student_id"] for s in students_data], EVAL_NUM_STUDENTS)
    
    actual_interactions = {sid: set() for sid in sample_student_ids}
    for interaction in interactions_data:
        if interaction["student_id"] in actual_interactions:
            actual_interactions[interaction["student_id"]].add(interaction["job_id"])

    metrics = {"precision@k": [], "recall@k": [], "ndcg@k": [], "mrr@k": []}
    max_skills_student = model.student_embedding_model.input_shape["student_skill_ids"][1]
    
    for sid in sample_student_ids:
        student = student_features_map[sid]
        student_skills_padded = student["skill_ids"] + [0] * (max_skills_student - len(student["skill_ids"]))
        
        student_query_vec = model.student_embedding_model.predict({
            "student_id": np.array([sid]),
            "student_faculty_id": np.array([student["faculty_id"]]),
            "student_skill_ids": np.array([student_skills_padded])
        }, verbose=0)

        _, recommended_jids = retrieval_index(student_query_vec)
        recommended_jids = recommended_jids.numpy()[0]
        
        actual_set = actual_interactions.get(sid, set())
        if not actual_set: continue
        
        hits = len(set(recommended_jids) & actual_set)
        relevance = [1 if jid in actual_set else 0 for jid in recommended_jids]
        
        metrics["precision@k"].append(hits / TOP_K)
        metrics["recall@k"].append(hits / len(actual_set))
        metrics["ndcg@k"].append(_ndcg_at_k(relevance, TOP_K))
        
        first_hit_idx = next((i for i, r in enumerate(relevance) if r == 1), -1)
        metrics["mrr@k"].append(1 / (first_hit_idx + 1) if first_hit_idx != -1 else 0)

    print(f"評価指標 (k={TOP_K}):")
    for name, values in metrics.items():
        mean_val = np.mean(values)
        print(f"  - 平均 {name.upper()}: {mean_val:.4f}")

    df_metrics = pd.DataFrame(metrics)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_metrics, showmeans=True)
    plt.title(f"Ranking Metrics Distribution (k={TOP_K}, n={EVAL_NUM_STUDENTS})")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- メイン処理 ---
if __name__ == "__main__":
    students_data, jobs_data, interactions_data_raw, vocab_size = generate_dummy_data()
    train_ds, student_features_map, job_features_map, max_s_s, max_s_j, interactions_data = prepare_datasets(students_data, jobs_data, interactions_data_raw)
    
    print("--- 3. モデルの構築開始 ---")
    bpr_model = TwoTowerBPRModel(vocab_size, max_s_s, max_s_j, jobs_data)
    _ = bpr_model(next(iter(train_ds))) # 重みを初期化
    bpr_model.student_embedding_model.summary(line_length=100)
    print("--- モデル構築完了 ---\n")
    
    print("--- 4. 学習開始 ---")
    bpr_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    steps_per_epoch = math.ceil(len(interactions_data) / BATCH_SIZE)
    bpr_model.fit(train_ds, epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch, verbose=1)
    print("--- 学習完了 ---\n")
    
    print("--- 5. 推薦インデックスの構築 ---")
    all_job_ids_np = np.array([j["job_id"] for j in jobs_data], dtype=np.int32)
    all_job_industry_ids_np = np.array([j.get("industry_id", 0) for j in jobs_data], dtype=np.int32)
    all_job_skills_padded = [j["required_skill_ids"] + [0] * (max_s_j - len(j["required_skill_ids"])) for j in jobs_data]
    all_job_skills_np = np.array(all_job_skills_padded, dtype=np.int32)

    job_embeddings = bpr_model.job_embedding_model.predict({
        "job_id": all_job_ids_np,
        "job_industry_id": all_job_industry_ids_np,
        "job_skill_ids": all_job_skills_np
    }, verbose=0)

    retrieval_index = keras_rs.layers.BruteForce(k=TOP_K)
    retrieval_index.index(tf.constant(job_embeddings), all_job_ids_np)
    print("--- インデックス構築完了 ---")
    
    evaluate_ranking(bpr_model, retrieval_index, student_features_map, students_data, interactions_data)
