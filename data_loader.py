import os
import io
import zipfile
import requests
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
CSV_FILE = os.path.join(DATA_DIR, "sms_spam.csv")


def download_dataset(force: bool = False) -> str:
    """
    UCI SMS Spam Collection 데이터셋을 다운로드하여 data/에 저장.

    @param {bool} force - True이면 기존 파일이 있어도 재다운로드
    @return {str} 저장된 원본 파일 경로
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_FILE) and not force:
        print(f"[skip] 이미 존재: {RAW_FILE}")
        return RAW_FILE

    print(f"[download] {DATA_URL}")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_DIR)

    print(f"[done] saved to {RAW_FILE}")
    return RAW_FILE


def load_dataframe() -> pd.DataFrame:
    """
    데이터를 DataFrame으로 로드. label: 0=ham, 1=spam.

    @return {pd.DataFrame} text, label 컬럼을 가진 DataFrame
    """
    if not os.path.exists(RAW_FILE):
        download_dataset()

    df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
    )
    df["label"] = (df["label"].str.lower() == "spam").astype(int)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    df = load_dataframe()
    print(df.head())
    print(f"\n총 {len(df)}건 | spam: {df.label.sum()} | ham: {(df.label == 0).sum()}")
