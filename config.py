"""Model config."""

model_save_name: str = "model.h5"
batch_size: int = 64
epochs: int = 30

# Number of epochs with no improvement before reducing the learning rate.
patience: int = 3

factor: float = 0.5  # Reduce the learning rate by a factor of 0.5.
min_lr: float = 0.00001  # Minimum learning rate.
