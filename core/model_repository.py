from easy_torchkit.src.classification import ClassificationModel
import torch
from typing import OrderedDict


class CNNClassifier(ClassificationModel):
    def __init__(
        self,
        in_channels: int = 1,
        kernel_size: int = 50,
        stride: int = 10,
        cnn_out_channels: int = 5,
        pooling_output: int = 5,
        hidden_size: int = 25,
        num_classes: int = 3,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            device=device,
            track_best_model=track_best_model,
            random_state=random_state,
        )

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            out_channels=cnn_out_channels,
                        ),
                    ),
                    ("tanh", torch.nn.Tanh()),
                    ("pooling", torch.nn.AdaptiveAvgPool1d(pooling_output)),
                    ("flatten", torch.nn.Flatten()),
                    ("classifier", torch.nn.Linear(hidden_size, num_classes)),
                ]
            )
        )

        self.to(self.device)


class LSTMClassifier(ClassificationModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 25,
        num_classes: int = 3,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        random_state: int | None = None,
    ):
        # 1. Initialize parent first
        super().__init__(
            device=device,
            track_best_model=track_best_model,
            random_state=random_state,
        )

        # 2. Define layers individually (Not in Sequential)
        # This ensures they are registered as sub-modules correctly
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

        # 3. Explicitly move the whole model to device
        self.to(self.device)

    def forward(
        self, x: torch.Tensor, *, output_layer: str | None = None
    ) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        lstm_out, (hn, cn) = self.lstm(x)

        # Use the last time step's output
        last_time_step = lstm_out[:, -1, :]

        out = torch.tanh(last_time_step)
        logits = self.fc(out)

        return logits


class DNNClassifier(ClassificationModel):
    def __init__(
        self,
        input_size: int = 1000,
        hidden_size: int = 25,
        num_classes: int = 3,
        device: torch.device = torch.device("cpu"),
        track_best_model: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            device=device,
            track_best_model=track_best_model,
            random_state=random_state,
        )

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    ("lin", torch.nn.Linear(input_size, hidden_size)),
                    ("tanh", torch.nn.Tanh()),
                    ("fc", torch.nn.Linear(hidden_size, num_classes)),
                ]
            )
        )

        self.to(self.device)
