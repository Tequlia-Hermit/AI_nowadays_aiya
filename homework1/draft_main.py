# my_experiment.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss

RESULTS_DIR = Path('results')
PLOTS_DIR = RESULTS_DIR / 'plots'
METRICS_DIR = RESULTS_DIR / 'metrics'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def plot_lr_loss_by_budget(
    lr_model, train_features, train_labels, val_features, val_labels,
    output_path=PLOTS_DIR / 'lr_loss_curve.png',
    records_path=METRICS_DIR / 'lr_loss_curve.csv',
):
    """
    对 LogisticRegression 按不同迭代预算独立重训，
    计算并打印 train / val log loss。

    注意：非终点预算独立重训，不是一次拟合的逐轮损失。
    """
    final_iterations = int(np.max(lr_model.n_iter_))
    budgets = [
            b for b in [
        1, 2, 3, 5, 8, 10, 15, 20, 30, 50,
        80, 100, 150, 200, 300, 500, 800,
        1000, 1500, 2000, 3000, 4000, 6000, 8000
        ]
        if b < final_iterations
    ]
    budgets.append(max(1, final_iterations))

    records = []
    for index, budget in enumerate(budgets):
        if index == len(budgets) - 1:
            model = lr_model
            converged = True
        else:
            # 非终点预算独立重训，不是一次拟合的逐轮损失
            model = clone(lr_model).set_params(max_iter=budget)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(train_features, train_labels)
            converged = not any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )

        train_loss = log_loss(
            train_labels, model.predict_proba(train_features),
            labels=model.classes_
        )
        val_loss = log_loss(
            val_labels, model.predict_proba(val_features),
            labels=model.classes_
        )
        records.append({
            'iteration_budget': budget,
            'actual_max_iterations': int(np.max(model.n_iter_)),
            'train_log_loss': train_loss,
            'validation_log_loss': val_loss,
            'converged': converged,
            'reused_best_model': model is lr_model,
        })
        print(
            f"迭代预算={budget} | Train log loss={train_loss:.6f} | "
            f"Validation log loss={val_loss:.6f} | 已收敛={converged}",
            flush=True,
        )

    results = pd.DataFrame(records)
    results.to_csv(records_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(results['iteration_budget'], results['train_log_loss'],
            marker='o', label='Training (measured)')
    ax.plot(results['iteration_budget'], results['validation_log_loss'],
            marker='s', label='Validation (measured)')
    ax.set_xscale('log')
    ax.set_xlabel('Iteration budget (log scale; independent fits)')
    ax.set_ylabel('Log loss (without regularization)')
    ax.set_title(f'Logistic Regression: measured loss by iteration budget\nC={lr_model.C}')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(
        0.5, 0.02,
        'Fixed hyperparameters; not an epoch trace. Last point reuses the best model.',
        ha='center', fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"损失图已保存至：{output_path}；曲线数据：{records_path}", flush=True)
    return fig




# ========== 1. 加载数据 ==========
def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

X_train, y_train, X_test_unlabeled = load_data()

# ========== 2. 文本清洗 ==========
def clean_text(text):
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        text = parts[1]
    else:
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if not re.match(r'^(From|Subject|Organization|Lines|Reply-To|NNTP-Posting-Host|Date|Message-ID|References):', line, re.IGNORECASE):
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

    lines = text.split('\n')
    lines = [line for line in lines if not line.strip().startswith('>')]
    text = '\n'.join(lines)

    text = re.split(r'\n--\s*\n', text)[0]
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_train_cleaned = [clean_text(t) for t in X_train]
X_test_cleaned = [clean_text(t) for t in X_test_unlabeled]

# ========== 3. 划分验证集 ==========
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_cleaned, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# ========== 4. TF-IDF 特征提取 ==========
vectorizer = TfidfVectorizer(
    max_features=100000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)
X_train_tfidf = vectorizer.fit_transform(X_train_sub)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test_cleaned)

# ========== 5. 训练与验证多个模型 ==========
models = {
    'SVM': SVC(kernel='linear', C=1.0, random_state=42),
    'NB': MultinomialNB(alpha=0.1),
    'LR': LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train_sub)
    y_val_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_val_pred)
    results[name] = acc
    print(f"\n===== {name} 验证集准确率: {acc:.4f} =====")
    #print(classification_report(y_val, y_val_pred))
    #print(confusion_matrix(y_val, y_val_pred))

# ========== 6. 选择最佳模型，重新在全部训练集上训练 ==========
best_model_name = max(results, key=results.get)
print(f"\n最佳模型: {best_model_name}, 验证集准确率: {results[best_model_name]:.4f}")

# 用全部训练数据重新拟合 TF-IDF 和模型
vectorizer_full = TfidfVectorizer(
    max_features=100000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X_train_full_tfidf = vectorizer_full.fit_transform(X_train_cleaned)
X_test_full_tfidf = vectorizer_full.transform(X_test_cleaned)

best_model = models[best_model_name]
best_model.fit(X_train_full_tfidf, y_train)
predictions = best_model.predict(X_test_full_tfidf)

# ========== 7. 保存预测结果 ==========
pd.DataFrame({
    'id': range(len(predictions)),
    'target': predictions
}).to_csv('predictions.csv', index=False)

print("\n预测结果已保存到 predictions.csv")

# ========== 独立打印 LR 的损失函数 ==========
print("\n--- 开始打印 LogisticRegression 的损失函数 ---")
plot_lr_loss_by_budget(
    models['LR'],
    X_train_tfidf, y_train_sub,
    X_val_tfidf, y_val,
)