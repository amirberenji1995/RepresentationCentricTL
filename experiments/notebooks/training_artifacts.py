from easy_torchkit.src.early_stopping import StoppingCriteria, EffectiveSet
from easy_torchkit.src.contracts.configurations import (
    EvaluationMetric,
    TrainingPhaseType,
)
from easy_torchkit.src.training_steps.supervised_training_step import (
    SupervisedTrainingStep,
)
from easy_torchkit.src.training_steps.siamese_training_step import (
    ContrastiveLoss,
    SiameseTrainingStep,
)
from easy_torchkit.src.training_steps.dynamic_bootstrapping import (
    DynamicBootstrappingTrainingStep,
)
from sklearn.metrics import accuracy_score
import torch

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
    "training_step": SupervisedTrainingStep(),
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
    "training_step": SupervisedTrainingStep(),
    "phase": TrainingPhaseType.fine_tuning,
    "print_every": 1000,
    "stopping_criteria": criteria_list,
}

training_params_step_dict = {
    "training": training_params_dict,
    "fine_tuning": {
        "supervised": fine_tuning_params_dict,
        "contrastive": {
            "epochs": 5,
            "val_size": 0.25,
            "loss_fn": ContrastiveLoss(margin=1.0),
            "optimizer": torch.optim.Adam,
            "training_step": SiameseTrainingStep(),
            "phase": TrainingPhaseType.fine_tuning,
            "print_every": 1000,
            "stopping_criteria": criteria_list,
        },
        "dynamic_bootstrapping": {
            "epochs": 1000,
            "val_size": 0.25,
            "metrics": [accuracy_metric],
            "loss_fn": torch.nn.CrossEntropyLoss(reduction="mean"),
            "optimizer": torch.optim.Adam,
            "training_step": DynamicBootstrappingTrainingStep(),
            "phase": TrainingPhaseType.fine_tuning,
            "print_every": 1000,
            "stopping_criteria": criteria_list,
        },
    },
}
