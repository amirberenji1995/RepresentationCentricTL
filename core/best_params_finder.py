import optuna
from optuna.samplers import GridSampler
from easy_torchkit.src.contracts.training_params import TrainingParams
from easy_torchkit.src.classification import ClassificationModel
from easy_torchkit.src.training_steps.supervised_training_step import (
    SupervisedTrainingStep,
)
from easy_torchkit.src.training_steps.siamese_training_step import SiameseTrainingStep
import torch
from typing import Literal, Dict, List, Any


def make_contrastive_pairs(
    x: torch.Tensor,
    y: torch.Tensor,
    pairs_per_sample: int = 1,
):
    """
    Returns:
        pair_x: [N_pairs, 2, input_dim]
        pair_y: [N_pairs] (1 = similar, 0 = dissimilar)
    """
    device = x.device
    pair_x = []
    pair_y = []

    n = x.size(0)

    for i in range(n):
        xi, yi = x[i], y[i]

        same = (y == yi).nonzero(as_tuple=False).flatten()
        diff = (y != yi).nonzero(as_tuple=False).flatten()

        same = same[same != i]

        if len(same) == 0 or len(diff) == 0:
            continue

        num_pos = max(1, pairs_per_sample // 2)
        num_neg = pairs_per_sample - num_pos

        pos_idx = same[torch.randint(len(same), (num_pos,), device=device)]
        neg_idx = diff[torch.randint(len(diff), (num_neg,), device=device)]

        for j in pos_idx:
            pair_x.append(torch.stack([xi, x[j]], dim=0))
            pair_y.append(1)

        for j in neg_idx:
            pair_x.append(torch.stack([xi, x[j]], dim=0))
            pair_y.append(0)

    pair_x = torch.stack(pair_x, dim=0)
    pair_y = torch.tensor(pair_y, device=device, dtype=torch.float32)

    return pair_x, pair_y


def tune_source_phase(
    model: ClassificationModel,
    dataset_name: Literal["mfpt", "cwru", "kaist"],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    search_space: Dict[str, Dict[str, List[float]]],
    training_params_dict: Dict[str, Any],
):
    """
    Finds best params for source training without redundant fitting.
    Returns: (best_params, best_model_object)
    """
    grid = search_space[dataset_name]
    trial_models = {}

    def objective(trial):
        lr = trial.suggest_categorical("lr", grid["lr"])
        batch_size = trial.suggest_categorical("batch_size", grid["batch_size"])

        training_params_dict.update(
            {
                "lr": lr,
                "batch_size": batch_size,
            }
        )

        training_params = TrainingParams(**training_params_dict)

        trial_model = model.copy(reset_history=True)

        print(
            f"DEBUG: Data shapes -> x_train: {train_x.shape}, y_train: {train_y.shape}"
        )

        trial_model.fit(train_x, train_y, training_params)

        trial_models[trial.number] = trial_model

        return trial_model.best_val_loss

    print(f"\n>>> Tuning Source Phase: {dataset_name.upper()}")
    study = optuna.create_study(direction="minimize", sampler=GridSampler(grid))
    study.optimize(objective)

    best_trial_num = study.best_trial.number
    best_model = trial_models[best_trial_num]
    best_model.recover_best_model()

    return study.best_params, best_model, study


def tune_fine_tuning_phase(
    model: ClassificationModel,
    target_name: Literal["mfpt", "cwru", "kaist"],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    search_space: Dict[str, Dict[str, List[float]]],
    fine_tuning_params_dict: Dict[str, Any],
):
    """
    Finds best params for fine-tuning without redundant fitting.
    Returns: (best_params, best_model_object)
    """
    grid = search_space[target_name]
    trial_models = {}

    def objective(trial):

        print(
            f"\n\n\n>>> Tuning Fine-Tuning Phase: {fine_tuning_params_dict['training_step']}\n\n\n"
        )

        lr = trial.suggest_categorical("lr", grid["lr"])
        batch_size = trial.suggest_categorical("batch_size", grid["batch_size"])
        if isinstance(fine_tuning_params_dict["training_step"], SupervisedTrainingStep):
            fine_tuning_params_dict.update(
                {
                    "lr": lr,
                    "batch_size": batch_size,
                }
            )

            fine_tuning_params = TrainingParams(**fine_tuning_params_dict)

            trial_model = model.copy(reset_history=True)
            trial_model.fit(train_x, train_y, fine_tuning_params)

            trial_models[trial.number] = trial_model
            return trial_model.best_val_loss
        elif isinstance(fine_tuning_params_dict["training_step"], SiameseTrainingStep):
            fine_tuning_params_dict.update(
                {
                    "lr": lr,
                    "batch_size": batch_size,
                }
            )

            pair_x, pair_y = make_contrastive_pairs(
                train_x,
                train_y,
                pairs_per_sample=fine_tuning_params_dict.get("pairs_per_sample", 1),
            )

            fine_tuning_params = TrainingParams(**fine_tuning_params_dict)

            trial_model = model.copy(reset_history=True)

            # IMPORTANT: fit on paired data
            trial_model.fit(pair_x, pair_y, fine_tuning_params)

            trial_models[trial.number] = trial_model
            return trial_model.best_val_loss
        else:
            raise NotImplementedError

    study = optuna.create_study(direction="minimize", sampler=GridSampler(grid))
    study.optimize(objective)

    best_trial_num = study.best_trial.number
    best_model = trial_models[best_trial_num]
    best_model.recover_best_model()

    return study.best_params, best_model, study
