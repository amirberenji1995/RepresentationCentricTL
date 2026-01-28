from easy_torchkit.src.early_stopping import StoppingCriteria, EffectiveSet
from easy_torchkit.src.configurations import EvaluationMetric, TrainingPhaseType
from sklearn.metrics import accuracy_score
import torch
from easy_torchkit.src.utils import supervised_step

criteria_list = [
    StoppingCriteria(
        metric_name="loss",
        effective_set=EffectiveSet.VAL,
        mode="min",
        patience=10,
        min_epoch=100,
        message="EARLY STOPPING! --- Validation loss did not decrease in last 10 epochs.",
    ),
]

accuracy_metric = EvaluationMetric(name="accuracy", function=accuracy_score)

training_params_dict = {
    "epochs": 1000,
    "val_size": 0.25,
    "metrics": [accuracy_metric],
    "loss_fn": torch.nn.CrossEntropyLoss(reduction="mean"),
    "optimizer": torch.optim.Adam,
    "training_step": supervised_step,
    "phase": TrainingPhaseType.training,
    "print_every": 1000,
    "stopping_criteria": criteria_list,
}


fine_tuning_params_dict = {
    "epochs": 1000,
    "val_size": 0.25,
    "metrics": [accuracy_metric],
    "loss_fn": torch.nn.CrossEntropyLoss(reduction="mean"),
    "optimizer": torch.optim.Adam,
    "training_step": supervised_step,
    "phase": TrainingPhaseType.fine_tuning,
    "print_every": 1000,
    "stopping_criteria": criteria_list,
}
