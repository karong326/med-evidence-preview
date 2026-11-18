"""Keep probability resolution independent of the model's compute precision."""
from torch import Tensor, nn


class Float32Sigmoid(nn.Module):
    """Avoid FP16 sigmoid rounding collapsing distinct high-relevance scores."""

    def forward(self, logits: Tensor) -> Tensor:
        return logits.float().sigmoid()
