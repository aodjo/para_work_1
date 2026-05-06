import os
import json
import pickle
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_dataframe
from utils import pick_device, print_device_info, EarlyStopping, plot_training_curves
from models.logistic import LogisticRegression

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def to_tensor(sparse_matrix) -> torch.Tensor:
    """
    scipy sparse matrix를 torch.Tensor로 변환.

    @param {scipy.sparse.spmatrix} sparse_matrix - 희소 행렬
    @return {torch.Tensor} float32 밀집 텐서
    """
    return torch.from_numpy(sparse_matrix.toarray().astype(np.float32))


def compute_metrics(y_true, y_pred) -> dict:
    """
    분류 성능 지표 계산.

    @param {array-like} y_true - 실제 라벨
    @param {array-like} y_pred - 예측 라벨
    @return {dict} accuracy, precision, recall, f1, confusion_matrix
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_loader(model, loader, device, threshold=0.5):
    """
    DataLoader 전체에 대해 모델을 평가.

    @param {nn.Module} model - 평가할 모델
    @param {DataLoader} loader - 테스트 DataLoader
    @param {torch.device} device - 디바이스
    @param {float} threshold - spam 판별 임계값
    @return {dict} 분류 성능 지표
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred.extend((probs >= threshold).astype(int).tolist())
            y_true.extend(yb.numpy().tolist())
    return compute_metrics(y_true, y_pred)


def train_tfidf_model(args, device, X_train_text, X_test_text, y_train, y_test):
    """
    TF-IDF 벡터화 후 로지스틱 회귀 모델 학습.

    @param {argparse.Namespace} args - 학습 하이퍼파라미터
    @param {torch.device} device - 학습 디바이스
    @param {np.ndarray} X_train_text - 학습 텍스트 배열
    @param {np.ndarray} X_test_text - 테스트 텍스트 배열
    @param {np.ndarray} y_train - 학습 라벨
    @param {np.ndarray} y_test - 테스트 라벨
    @return {tuple} (model, test_loader, history)
    """
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=args.max_features, min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    input_dim = X_train.shape[1]
    print(f"[tfidf] feature_dim={input_dim}")

    X_train_t, X_test_t = to_tensor(X_train), to_tensor(X_test)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.float32))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size)

    model = LogisticRegression(input_dim).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    early_stop = EarlyStopping(patience=args.patience, mode="max")
    history = {"epoch": [], "train_loss": [], "val_f1": [], "val_accuracy": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        metrics = evaluate_loader(model, test_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(metrics["f1"])
        history["val_accuracy"].append(metrics["accuracy"])

        print(
            f"[epoch {epoch:02d}] loss={train_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

        if early_stop(metrics["f1"], model):
            print(f"[early stop] epoch {epoch}, best f1={early_stop.best_score:.4f}")
            break

    early_stop.restore_best(model)

    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "model_type": "logistic"},
               os.path.join(ARTIFACTS_DIR, "logistic.pt"))
    with open(os.path.join(ARTIFACTS_DIR, "logistic_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    return model, test_loader, history


def main():
    parser = argparse.ArgumentParser()
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
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = pick_device()
    print_device_info(device)

    df = load_dataframe()
    print(f"[data] total={len(df)}, spam={df.label.sum()}, ham={(df.label==0).sum()}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values,
        test_size=0.2, random_state=args.seed, stratify=df["label"].values,
    )

    print(f"\n{'='*60}")
    print(f"  Training: LOGISTIC")
    print(f"{'='*60}\n")

    model, test_loader, history = train_tfidf_model(
        args, device, X_train_text, X_test_text, y_train, y_test)

    final = evaluate_loader(model, test_loader, device)
    print(f"\n=== LOGISTIC Final Test Metrics ===")
    for k, v in final.items():
        print(f"  {k}: {v}")

    plot_training_curves(
        history,
        save_path=os.path.join(PLOTS_DIR, "logistic_curve.png"),
        title="LOGISTIC",
    )

    metrics_path = os.path.join(ARTIFACTS_DIR, "logistic_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n[saved] artifacts/logistic.pt, plots/logistic_curve.png")


if __name__ == "__main__":
    main()