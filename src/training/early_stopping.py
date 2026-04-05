"""
Early stopping callback.

Usage:
    from src.training import EarlyStopping

    early_stopping = EarlyStopping(patience=10, mode="min")

    for epoch in range(epochs):
        val_loss = validate(...)
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
"""


class EarlyStopping:
    """
    Early stopping callback to halt training when a metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: "min" (lower is better, e.g. loss) or "max" (higher is better, e.g. accuracy).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current epoch metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def reset(self):
        """Reset state — useful when reusing across multiple training runs."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
