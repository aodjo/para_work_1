import os
import json
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data_loader import load_dataframe
from utils import pick_device, print_device_info, EarlyStopping
from models.logistic import LogisticRegression

import matplotlib.pyplot as plt

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def to_tensor(sparse_matrix) -> torch.Tensor:
    """
    scipy 희소 행렬을 밀집 텐서로 변환.

    @param {scipy.sparse.spmatrix} sparse_matrix - TF-IDF 희소 행렬
    @return {torch.Tensor} float32 밀집 텐서
    """
    return torch.from_numpy(sparse_matrix.toarray().astype(np.float32))


def evaluate(model, loader, device):
    """
    DataLoader 전체에 대해 모델을 평가.

    @param {nn.Module} model - 평가할 모델
    @param {DataLoader} loader - 테스트 DataLoader
    @param {torch.device} device - 디바이스
    @return {dict} accuracy, precision, recall, f1 지표
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            y_pred.extend((probs >= 0.5).astype(int).tolist())
            y_true.extend(yb.numpy().tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_single(X_train_text, y_train, X_test_text, y_test, device, args):
    """
    주어진 학습 데이터로 로지스틱 모델을 학습하고 테스트 메트릭을 반환.

    @param {np.ndarray} X_train_text - 학습 텍스트 배열
    @param {np.ndarray} y_train - 학습 라벨
    @param {np.ndarray} X_test_text - 테스트 텍스트 배열
    @param {np.ndarray} y_test - 테스트 라벨
    @param {torch.device} device - 학습 디바이스
    @param {argparse.Namespace} args - 하이퍼파라미터
    @return {dict} accuracy, precision, recall, f1 지표
    """
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=args.max_features, min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    input_dim = X_train.shape[1]

    train_loader = DataLoader(
        TensorDataset(to_tensor(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(to_tensor(X_test), torch.from_numpy(y_test.astype(np.float32))),
        batch_size=args.batch_size,
    )

    model = LogisticRegression(input_dim).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    early_stop = EarlyStopping(patience=args.patience, mode="max")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, test_loader, device)
        if early_stop(metrics["f1"], model):
            break

    early_stop.restore_best(model)
    return evaluate(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--repeat", type=int, default=3, help="각 비율당 반복 횟수 (평균 산출)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = pick_device()
    print_device_info(device)

    df = load_dataframe()
    X_all = df["text"].values
    y_all = df["label"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all,
    )

    print(f"[data] train={len(X_train_full)}, test={len(X_test)}")
    print(f"[실험] fractions={args.fractions}, repeat={args.repeat}\n")

    results = []

    for frac in args.fractions:
        n_samples = int(len(X_train_full) * frac)
        metrics_list = []

        for r in range(args.repeat):
            seed_r = args.seed + r
            if frac < 1.0:
                X_sub, _, y_sub, _ = train_test_split(
                    X_train_full, y_train_full,
                    train_size=frac, random_state=seed_r, stratify=y_train_full,
                )
            else:
                X_sub, y_sub = X_train_full, y_train_full

            torch.manual_seed(seed_r)
            m = train_single(X_sub, y_sub, X_test, y_test, device, args)
            metrics_list.append(m)

        # 평균 및 표준편차
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        std = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0]}

        results.append({
            "fraction": frac,
            "n_samples": n_samples,
            "avg": avg,
            "std": std,
        })

        print(
            f"[{frac*100:5.1f}% | {n_samples:5d}건] "
            f"F1={avg['f1']:.4f}±{std['f1']:.4f}  "
            f"Acc={avg['accuracy']:.4f}±{std['accuracy']:.4f}  "
            f"Prec={avg['precision']:.4f}  Recall={avg['recall']:.4f}"
        )

    # ─── 결과 저장 ─────────────────────────────────────────────
    results_path = os.path.join(ARTIFACTS_DIR, "data_size_experiment.json")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {results_path}")

    # ─── 그래프 ───────────────────────────────────────────────
    fracs = [r["fraction"] * 100 for r in results]
    samples = [r["n_samples"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: F1 / Accuracy
    for key, color, marker in [("f1", "#e74c3c", "o"), ("accuracy", "#3498db", "s")]:
        means = [r["avg"][key] for r in results]
        stds = [r["std"][key] for r in results]
        axes[0].errorbar(fracs, means, yerr=stds, color=color, marker=marker,
                         markersize=6, capsize=4, linewidth=2, label=key.upper())
    axes[0].set_xlabel("Training Data (%)", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("Data Size vs Performance", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(fracs)

    # 오른쪽: Precision / Recall
    for key, color, marker in [("precision", "#2ecc71", "^"), ("recall", "#f39c12", "D")]:
        means = [r["avg"][key] for r in results]
        stds = [r["std"][key] for r in results]
        axes[1].errorbar(fracs, means, yerr=stds, color=color, marker=marker,
                         markersize=6, capsize=4, linewidth=2, label=key.capitalize())
    axes[1].set_xlabel("Training Data (%)", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Data Size vs Precision / Recall", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(fracs)

    # 상단에 샘플 수 표시
    for ax in axes:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(fracs)
        ax2.set_xticklabels([f"{n}" for n in samples], fontsize=8, color="gray")
        ax2.set_xlabel("N samples", fontsize=9, color="gray")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "data_size_experiment.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[plot] {plot_path}")


if __name__ == "__main__":
    main()
