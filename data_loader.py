"""한국어 SMS 스팸 데이터셋 로더 (meal-bbang/Korean_message)."""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILE = os.path.join(DATA_DIR, "korean_message.csv")


def download_dataset(force: bool = False) -> str:
    """
    HuggingFace에서 한국어 메시지 데이터셋을 다운로드하여 CSV로 저장.

    @param {bool} force - True이면 기존 파일이 있어도 재다운로드
    @return {str} 저장된 CSV 파일 경로
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_FILE) and not force:
        print(f"[skip] 이미 존재: {CSV_FILE}")
        return CSV_FILE

    from datasets import load_dataset

    print("[download] meal-bbang/Korean_message from HuggingFace")
    ds = load_dataset("meal-bbang/Korean_message", split="train")
    df = ds.to_pandas()

    # class 1(일반), 3(택배알림) → ham(0) / class 2(피싱) → spam(1)
    df = df[["content", "class"]].rename(columns={"content": "text", "class": "label"})
    df["label"] = df["label"].map({1: 0, 3: 0, 2: 1})
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    print(f"[done] saved to {CSV_FILE} ({len(df)}건)")
    return CSV_FILE


def load_dataframe() -> pd.DataFrame:
    """
    데이터를 DataFrame으로 로드. label: 0=ham, 1=spam.

    @return {pd.DataFrame} text, label 컬럼을 가진 DataFrame
    """
    if not os.path.exists(CSV_FILE):
        download_dataset()

    df = pd.read_csv(CSV_FILE, encoding="utf-8")
    return df


if __name__ == "__main__":
    df = load_dataframe()
    print(df.head(10))
    print(f"\n총 {len(df)}건 | spam: {df.label.sum()} | ham: {(df.label == 0).sum()}")
