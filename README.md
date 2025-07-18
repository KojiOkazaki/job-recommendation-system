# 就活支援レコメンドシステム・デモ

## 概要

本プロジェクトは、学生と求人のプロフィール（特徴量）を活用した、就活支援レコメンドシステムのデモンストレーションです。

TensorFlow(Keras)を使用し、学生と求人の特徴から埋め込みベクトルを学習する**Two-Towerモデル**を構築します。学習には、ランキングタスクに適した**BPR (Bayesian Personalized Ranking) 損失**を採用し、各学生に最適な求人を推薦します。

---

## アーキテクチャ

本システムは、「Two-Tower（双対の塔）」モデルを採用しています。

* **学生タワー (Query Tower)**: 学生ID、学部、スキルセットなどの特徴を入力とし、「学生ベクトル」を生成します。
* **求人タワー (Candidate Tower)**: 求人ID、業界、必須スキルなどの特徴を入力とし、「求人ベクトル」を生成します。

学習を通じて、関連性の高い「学生」と「求人」のベクトルが、ベクトル空間上で近傍に配置されるようにモデルの重みを更新します。推薦時には、特定の学生ベクトルと最も近傍にある求人ベクトル群を探索し、ランキング形式で提示します。

---

## 主な特徴

-   **Two-Towerモデル**: 学生と求人の特徴量を効率的に学習し、スケーラブルな推薦を実現します。
-   **BPR損失**: 観測されたインタラクション（ポジティブサンプル）と、観測されていないインタラクション（ネガティブサンプル）のペアを考慮し、推薦順位を直接的に最適化します。
-   **特徴量エンジニアリング**: 学部、スキル、業界などのカテゴリカルな特徴量を`Embedding`レイヤーでベクトル化し、豊かな表現を学習します。
-   **ランキング評価**: 推薦の質を多角的に評価するため、複数の標準的なランキング指標を導入しています。
    -   **Precision@k / Recall@k**: 上位k件の推薦リストにおける適合率と再現率。
    -   **NDCG@k (Normalized Discounted Cumulative Gain)**: 推薦リストの順序の適切性を評価する指標。関連性の高い項目が上位にあるほど高評価となります。
    -   **MRR@k (Mean Reciprocal Rank)**: 適合アイテムがリストの何番目に提示されたかを評価する指標。

---

## 実行結果の例

`python main.py` を実行すると、学習のログが表示された後、最終的に評価結果のサマリーと、ランキング指標の分布を示すボックスプロットが出力されます。これにより、モデルの性能を視覚的に確認できます。

---

## セットアップと実行手順

### 1. リポジトリのクローン
まず、プロジェクトをローカル環境にコピーします。
```bash
git clone [https://github.com/your-username/job-recommendation-system.git](https://github.com/your-username/job-recommendation-system.git)
cd job-recommendation-system
※ your-username の部分は、ご自身のGitHubユーザー名に置き換えてください。

2. 仮想環境の構築（推奨）
プロジェクトごとに環境を分離し、依存関係の競合を回避します。

Bash

# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate
# Windowsの場合は: venv\Scripts\activate
3. 依存ライブラリのインストール
requirements.txtファイルに基づき、必要なライブラリを一括でインストールします。

Bash

pip install -r requirements.txt
4. スクリプトの実行
メインスクリプトを実行します。ダミーデータの生成からモデルの学習、評価までが自動的に行われます。

Bash

python main.py
プロジェクト構成
.
├── main.py              # データ生成、モデル構築、学習、評価の全ロジックを含むメインスクリプト
├── requirements.txt       # プロジェクトの依存ライブラリリスト
└── README.md              # 本ドキュメント
main.pyの主要な関数:

generate_dummy_data(): デモンストレーション用のダミーデータを生成します。

prepare_datasets(): tf.data.Datasetパイプラインを構築します。

TwoTowerBPRModel: KerasのModelクラスを継承したモデル本体です。

evaluate_ranking(): 各種ランキング評価指標を算出し、結果を可視化します。

ライセンス
本プロジェクトは MIT License の下で公開されています。
