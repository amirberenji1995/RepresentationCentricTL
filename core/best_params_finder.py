import optuna
from optuna.samplers import GridSampler
import torch
import torch.nn.functional as F
from typing import Literal, Dict, List, Any
from easy_torchkit.src.contracts.training_params import TrainingParams
from easy_torchkit.src.classification import ClassificationModel
from easy_torchkit.src.training_steps.supervised_training_step import (
    SupervisedTrainingStep,
)
from easy_torchkit.src.training_steps.siamese_training_step import SiameseTrainingStep
from easy_torchkit.src.training_steps.dynamic_bootstrapping import (
    DynamicBootstrappingTrainingStep,
)
from .utils import make_contrastive_pairs


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
    fine_tuning_style_search_space: Dict | None = None,
    gt_label_recovery_rate: float | None = None,
):
    """
    Finds best params for fine-tuning without redundant fitting.
    Returns: (best_params, best_model_object)
    """
    grid = search_space[target_name].copy()
    if fine_tuning_style_search_space:
        grid.update(fine_tuning_style_search_space)
    trial_models = {}

    def objective(trial):
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
            pairs_per_sample = trial.suggest_categorical(
                "pairs_per_sample",
                fine_tuning_style_search_space["pairs_per_sample"],
            )
            fine_tuning_params_dict.update(
                {
                    "lr": lr,
                    "batch_size": batch_size,
                }
            )

            pair_x, pair_y = make_contrastive_pairs(
                train_x, train_y, pairs_per_sample=pairs_per_sample
            )

            fine_tuning_params = TrainingParams(**fine_tuning_params_dict)

            trial_model = model.copy(reset_history=True)

            # IMPORTANT: fit on paired data
            trial_model.fit(pair_x, pair_y, fine_tuning_params)

            trial_models[trial.number] = trial_model
            return trial_model.best_val_loss

        elif isinstance(
            fine_tuning_params_dict["training_step"], DynamicBootstrappingTrainingStep
        ):
            fine_tuning_params_dict.update(
                {
                    "lr": lr,
                    "batch_size": batch_size,
                    "training_step": DynamicBootstrappingTrainingStep(
                        warmup_epochs=trial.suggest_categorical(
                            "warmup_epochs",
                            fine_tuning_style_search_space["warmup_epochs"],
                        ),
                        bmm_iters=trial.suggest_categorical(
                            "bmm_iters",
                            fine_tuning_style_search_space["bmm_iters"],
                        ),
                    ),
                }
            )

            fine_tuning_params = TrainingParams(**fine_tuning_params_dict)
            trial_model = model.copy(reset_history=True)

            # 1. Get initial refurbished labels (model predictions)
            train_y_predicted = (
                F.softmax(trial_model.forward(train_x), dim=1).argmax(dim=1).detach()
            )

            # 2. Setup splitting
            num_samples = train_x.size(0)
            split_point = int(num_samples * (1 - fine_tuning_params.val_size))

            gen = torch.Generator(device="cpu").manual_seed(model.random_state)
            indices = torch.randperm(num_samples, generator=gen).to(train_x.device)

            # 3. Create the standard splits
            x_train_split = train_x[indices[:split_point]]
            y_train_refurbished = train_y_predicted[indices[:split_point]]

            x_val_split = train_x[indices[split_point:]]
            y_val_original = train_y[indices[split_point:]]

            # --- GT LABEL RECOVERY LOGIC ---
            if gt_label_recovery_rate != 0.0:
                # Determine which training samples to recover from ground truth
                num_train = x_train_split.size(0)
                # Generate a mask (on CPU first to match generator)
                recovery_mask = (
                    torch.rand(num_train, generator=gen) < gt_label_recovery_rate
                )
                recovery_mask = recovery_mask.to(train_x.device)

                # Get the ground truth labels for the training split
                y_train_gt = train_y[indices[:split_point]]

                # Replace refurbished with GT where mask is True
                y_train_refurbished = torch.where(
                    recovery_mask, y_train_gt, y_train_refurbished
                )
            trial_model.fit(
                x_train_split,
                y_train_refurbished,
                fine_tuning_params,
                eval_set=(x_val_split, y_val_original),
            )

            trial_models[trial.number] = trial_model
            return trial_model.best_val_loss

        else:
            raise NotImplementedError

    study = optuna.create_study(direction="minimize", sampler=GridSampler(grid))
    study.optimize(objective)

    best_trial_num = study.best_trial.number
    best_model = trial_models[best_trial_num]
    best_model.recover_best_model()
    print(f"\n\n\n>>> DEBGUG: Best Params Found: {study.best_params}\n\n\n")
    return study.best_params, best_model, study
