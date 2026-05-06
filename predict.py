"""저장된 Logistic Regression 모델로 새로운 메시지를 분류."""
import os
import pickle
import argparse

import numpy as np
import torch

from utils import pick_device
from models.logistic import LogisticRegression

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def load_model(device: torch.device):
    """
    저장된 체크포인트에서 모델을 로드.

    @param {torch.device} device - 모델을 올릴 디바이스
    @return {LogisticRegression} 평가 모드로 설정된 모델
    """
    ckpt = torch.load(os.path.join(ARTIFACTS_DIR, "logistic.pt"), map_location="cpu")
    model = LogisticRegression(ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(messages: list[str], device: torch.device | None = None):
    """
    메시지 리스트를 spam/ham으로 분류.

    @param {list[str]} messages - 분류할 문자 메시지 리스트
    @param {torch.device | None} device - 추론 디바이스 (None이면 자동 선택)
    @return {list[dict]} text, label, spam_prob 키를 가진 딕셔너리 리스트
    """
    if device is None:
        device = pick_device()
    model = load_model(device)

    with open(os.path.join(ARTIFACTS_DIR, "logistic_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(messages).toarray().astype(np.float32)
    X_t = torch.from_numpy(X).to(device)
    logits = model(X_t)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = (probs >= 0.5).astype(int)

    return [
        {"text": t, "label": "spam" if l == 1 else "ham", "spam_prob": float(p)}
        for t, l, p in zip(messages, labels, probs)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("messages", nargs="*")
    args = parser.parse_args()

    if args.messages:
        messages = [" ".join(args.messages)]
    else:
        messages = [
            "Hey, are we still meeting for lunch tomorrow?",
            "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!",
            "URGENT! Your account has been suspended. Verify at http://bit.ly/xx",
            "Don't forget to bring the documents to the meeting.",
        ]

    device = pick_device()
    print(f"[device] {device}")

    results = predict(messages, device=device)
    for r in results:
        flag = "[SPAM]" if r["label"] == "spam" else "[ HAM]"
        print(f"{flag} p={r['spam_prob']:.3f}  {r['text']}")


if __name__ == "__main__":
    main()
