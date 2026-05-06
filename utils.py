import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def pick_device() -> torch.device:
    """
    사용 가능한 디바이스를 우선순위에 따라 선택 (XPU → CUDA → CPU).

    @return {torch.device} 선택된 디바이스
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info(device: torch.device):
    """
    선택된 디바이스 정보를 출력.

    @param {torch.device} device - 출력할 디바이스
    """
    print(f"[device] {device}")
    if device.type == "xpu":
        try:
            print(f"[device] Intel GPU: {torch.xpu.get_device_name(0)}")
        except Exception:
            pass
    elif device.type == "cuda":
        print(f"[device] NVIDIA GPU: {torch.cuda.get_device_name(0)}")


class EarlyStopping:
    """검증 지표가 개선되지 않으면 학습을 조기 종료."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        """
        @param {int} patience - 개선 없이 허용할 에폭 수
        @param {float} min_delta - 개선으로 인정할 최소 변화량
        @param {str} mode - "max"이면 증가를 개선으로, "min"이면 감소를 개선으로 판단
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        에폭마다 호출하여 조기 종료 여부 판단.

        @param {float} score - 현재 에폭의 검증 점수
        @param {nn.Module} model - 현재 모델 (최적 상태 저장용)
        @return {bool} True이면 학습 중단
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best(self, model: nn.Module):
        """
        최적 성능 시점의 가중치를 모델에 복원.

        @param {nn.Module} model - 가중치를 복원할 모델
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def plot_training_curves(history: dict, save_path: str, title: str = ""):
    """
    학습 곡선(loss, F1, accuracy)을 이미지로 저장.

    @param {dict} history - epoch, train_loss, val_f1, val_accuracy 키를 가진 딕셔너리
    @param {str} save_path - 저장할 이미지 경로
    @param {str} title - 그래프 제목 접두사
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_f1"], "r-o", markersize=3, label="Val F1")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], "g-s", markersize=3, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{title} - Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved: {save_path}")
