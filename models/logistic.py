"""Logistic Regression (baseline)."""
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """이진 분류를 위한 로지스틱 회귀 모델."""

    def __init__(self, input_dim: int):
        """
        모델 초기화.

        @param {int} input_dim - 입력 피처 차원 수
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 수행.

        @param {torch.Tensor} x - 입력 텐서 (batch_size, input_dim)
        @return {torch.Tensor} 로짓 값 (batch_size,)
        """
        return self.linear(x).squeeze(-1)
