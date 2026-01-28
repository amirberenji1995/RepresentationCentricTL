### --- Setting the base directory ---
import sys
import os
import argparse
import time
import numpy as np
import torch
import gc
from joblib import Parallel, delayed

project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

### --- Imports ---
from core.utils import (
    device_recognizer,
    datasets_loader,
    label_unifier,
    signal_target_decleration,
    label_encoder,
    train_test_splitter,
    extract_study_data,
    p_subsampler_torch,
    load_best_params_lookup,  # Assuming this exists in your utils
    Routine,
    BestParams,
)
from core.routine_registery import ROUTINE_REGISTERY
from core.best_params_finder import tune_fine_tuning_phase
from easy_torchkit.src.configurations import TrainingParams
from experiments.notebooks.training_artifacts import (
    training_params_dict,
    fine_tuning_params_dict,
)

# ---------------------------------------------------------
# 1. ARGPARSE FOR TERMINAL CONTROL
# ---------------------------------------------------------
# Detect default device
default_device = device_recognizer()

parser = argparse.ArgumentParser(
    description="Tune Fine-Tuning Params with Fixed Source"
)
parser.add_argument(
    "--subsample",
    type=float,
    default=None,
    help="Subsampling factor for fine-tuning (e.g., 0.1).",
)
parser.add_argument(
    "--workers",
    type=int,
    default=3,
    help="Number of routines to run in parallel.",
)
parser.add_argument(
    "--params_file",
    type=str,
    default="best_params.jsonl",
    help="Source of the training hyperparameters.",
)
parser.add_argument(
    "--device",
    type=str,
    default=str(default_device),
    help="Device to use (e.g., 'cuda:0', 'cuda:1', 'cpu').",
)

args = parser.parse_args()

# Override the global device with user input
device = torch.device(args.device)
fine_tuning_set_subsampling_factor = args.subsample

# ---------------------------------------------------------
# 2. INITIAL SETTINGS & DATA LOADING
# ---------------------------------------------------------
datasets = datasets_loader()
mfpt_data, cwru_data, kaist_data = label_unifier(
    datasets["mfpt"], datasets["cwru"], datasets["kaist"]
)

x_mfpt, y_mfpt, x_cwru, y_cwru, x_kaist, y_kaist = signal_target_decleration(
    mfpt_data, cwru_data, kaist_data
)
y_mfpt_encoded, y_cwru_encoded, y_kaist_encoded = label_encoder(
    [y_mfpt, y_cwru, y_kaist]
)

experiment_data = {
    "mfpt": {"x": x_mfpt, "y": y_mfpt_encoded},
    "cwru": {"x": x_cwru, "y": y_cwru_encoded},
    "kaist": {"x": x_kaist, "y": y_kaist_encoded},
}

# Load pre-existing best parameters
best_params_lookup = load_best_params_lookup(args.params_file)


# ---------------------------------------------------------
# 3. THE WORKER FUNCTION
# ---------------------------------------------------------
def execute_routine(routine_key):
    print(f"\n>>> Starting Routine: {routine_key}")

    current_ft_params = {}
    best_models_in_routine = {}

    random_state = np.random.randint(1, 1000)
    import copy

    routine = Routine(**copy.deepcopy(ROUTINE_REGISTERY[routine_key]))

    # Identify existing training params for this routine
    if routine.description not in best_params_lookup:
        print(f"Skipping {routine_key}: Description not found in {args.params_file}")
        return None

    # Retrieve the fixed training params
    fixed_training_params = best_params_lookup[routine.description][
        "training_best_params"
    ]

    raw_splits = train_test_splitter(
        experiment_data, test_size=0.4, random_state=random_state
    )

    start_time = time.time()

    try:
        # --- PHASE 1: Source Training (Fixed Hyperparams) ---
        for ds_name in ["mfpt", "cwru", "kaist"]:
            x_train = routine.apply_preprocessing(
                raw_splits[ds_name]["train_x"], ds_name
            )
            y_train = torch.tensor(raw_splits[ds_name]["train_y"], dtype=torch.long).to(
                device
            )
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)

            # 1. Clean and Setup Model Args
            model_params = routine.model[1].get(ds_name, {}).copy()
            win_len_val = model_params.pop("win_len", None)
            model_params.pop("hop_len", None)

            model_kwargs = {
                "device": device,
                "random_state": random_state,
                **model_params,
            }

            model_type = routine.model[0].__name__
            if model_type == "LSTMClassifier":
                if win_len_val:
                    model_kwargs["input_size"] = win_len_val
            elif model_type == "DNNClassifier":
                model_kwargs["input_size"] = x_train_tensor.size()[-1]

            model_instance = routine.model[0](**model_kwargs)

            # 2. Apply FIXED Hyperparameters from Lookup
            src_hp = fixed_training_params[ds_name]
            t_params = TrainingParams(
                **{
                    **training_params_dict,
                    "lr": src_hp["lr"],
                    "batch_size": src_hp["batch_size"],
                }
            )

            # 3. Train without tuning
            model_instance.fit(x_train_tensor, y_train, t_params)
            model_instance.recover_best_model()
            best_models_in_routine[ds_name] = model_instance

        # --- PHASE 2: Fine-Tuning (Hyperparameter Tuning) ---
        for src_name in ["mfpt", "cwru", "kaist"]:
            current_ft_params[src_name] = {}
            targets = [ds for ds in ["mfpt", "cwru", "kaist"] if ds != src_name]

            for tgt_name in targets:
                x_target = routine.apply_preprocessing(
                    raw_splits[tgt_name]["train_x"], tgt_name, src_name
                )
                y_target = torch.tensor(
                    raw_splits[tgt_name]["train_y"], dtype=torch.long
                ).to(device)

                x_target_tensor = torch.tensor(x_target, dtype=torch.float32).to(device)

                if isinstance(fine_tuning_set_subsampling_factor, float):
                    x_target_tensor, y_target = p_subsampler_torch(
                        x=x_target_tensor,
                        y=y_target,
                        rnd_state=random_state,
                        p=fine_tuning_set_subsampling_factor,
                    )

                # Tune ONLY the fine-tuning phase
                ft_params, _, ft_study = tune_fine_tuning_phase(
                    best_models_in_routine[src_name],
                    tgt_name,
                    x_target_tensor,
                    y_target,
                    routine.training_best_params_ranges,
                    fine_tuning_params_dict,
                )
                current_ft_params[src_name][tgt_name] = ft_params

        # Export Combined Results
        bp = BestParams(
            description=routine.description,
            training_best_params=fixed_training_params,  # Re-using the fixed ones
            fine_tuning_best_params=current_ft_params,  # New ones found today
            random_state=random_state,
            training_study_details=None,  # No study performed for Phase 1
            fine_tuning_study_details=extract_study_data(ft_study),
        )

        p_suffix = (
            f"_p_{str(fine_tuning_set_subsampling_factor).replace('.', '')}"
            if fine_tuning_set_subsampling_factor is not None
            else ""
        )

        bp.log_to_jsonl(
            log_file=f"best_params{p_suffix}.jsonl",
            exclude=["training_study_details", "fine_tuning_study_details"],
        )

    finally:
        del best_models_in_routine, routine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    print(f"DONE: {routine_key} | Time: {time.time() - start_time:.2f}s")
    return routine_key


# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    absolute_start_time = time.time()
    routine_keys = list(ROUTINE_REGISTERY.keys())

    print(f"Running FT Tuning | Workers: {args.workers} | Device: {device}")
    print(f"Using source params from: {args.params_file}")

    Parallel(n_jobs=args.workers, backend="loky")(
        delayed(execute_routine)(rk) for rk in routine_keys
    )

    print("\n" + "#" * 60)
    print(f"TOTAL EXECUTION TIME: {time.time() - absolute_start_time:.2f} seconds")
    print("#" * 60)
