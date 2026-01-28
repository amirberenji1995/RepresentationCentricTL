### --- Setting the base directory ---
import sys
import os
import argparse
import time
import numpy as np
import torch
import gc  # Added for garbage collection
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
    Routine,
    BestParams,
)
from core.routine_registery import ROUTINE_REGISTERY
from core.best_params_finder import tune_source_phase, tune_fine_tuning_phase
from experiments.notebooks.training_artifacts import (
    training_params_dict,
    fine_tuning_params_dict,
)

# ---------------------------------------------------------
# 1. ARGPARSE FOR TERMINAL CONTROL
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning Routines")
parser.add_argument(
    "--subsample",
    type=float,
    default=None,
    help="Subsampling factor for fine-tuning (e.g., 0.1 for 10%%). Default is None.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=3,
    help="Number of routines to run in parallel. Default is 3.",
)
args = parser.parse_args()

fine_tuning_set_subsampling_factor = args.subsample

# ---------------------------------------------------------
# 2. INITIAL SETTINGS (Shared across workers)
# ---------------------------------------------------------
device = device_recognizer()
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


# ---------------------------------------------------------
# 3. THE WORKER FUNCTION
# ---------------------------------------------------------
def execute_routine(routine_key):
    """Encapsulates the logic for a single routine."""
    print(f"\n>>> Starting Routine: {routine_key}")

    current_training_params = {}
    current_ft_params = {}
    best_models_in_routine = {}

    random_state = np.random.randint(1, 1000)
    import copy

    routine = Routine(**copy.deepcopy(ROUTINE_REGISTERY[routine_key]))

    raw_splits = train_test_splitter(
        experiment_data, test_size=0.4, random_state=random_state
    )

    start_time = time.time()

    try:
        # --- PHASE 1: Source Training ---
        for ds_name in ["mfpt", "cwru", "kaist"]:
            x_train = routine.apply_preprocessing(
                raw_splits[ds_name]["train_x"], ds_name
            )
            y_train = torch.tensor(raw_splits[ds_name]["train_y"], dtype=torch.long).to(
                device
            )
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)

            model_kwargs = routine.model[1].get(ds_name, {}).copy()

            if routine.model[0].__name__ == "LSTMClassifier":
                win_len_val = model_kwargs.pop("win_len", None)
                model_kwargs.pop("hop_len", None)
                if win_len_val is not None:
                    model_kwargs["input_size"] = win_len_val

            model_instance = routine.model[0](
                device=device, random_state=random_state, **model_kwargs
            )

            params, model, training_study = tune_source_phase(
                model_instance,
                ds_name,
                x_train_tensor,
                y_train,
                routine.training_best_params_ranges,
                training_params_dict,
            )
            current_training_params[ds_name] = params
            best_models_in_routine[ds_name] = model

        # --- PHASE 2: Fine-Tuning ---
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

                ft_params, _, fine_tuning_study = tune_fine_tuning_phase(
                    best_models_in_routine[src_name],
                    tgt_name,
                    x_target_tensor,
                    y_target,
                    routine.training_best_params_ranges,
                    fine_tuning_params_dict,
                )
                current_ft_params[src_name][tgt_name] = ft_params

        # Export Results
        bp = BestParams(
            description=routine.description,
            training_best_params=current_training_params,
            fine_tuning_best_params=current_ft_params,
            random_state=random_state,
            training_study_details=extract_study_data(training_study),
            fine_tuning_study_details=extract_study_data(fine_tuning_study),
        )

        p_suffix = (
            f"_p_{str(fine_tuning_set_subsampling_factor).replace('.', '')}"
            if fine_tuning_set_subsampling_factor
            else ""
        )
        bp.log_to_jsonl(
            log_file=f"best_params{p_suffix}.jsonl",
            exclude=["training_study_details", "fine_tuning_study_details"],
        )

    finally:
        # --- CLEANUP BLOCK ---
        # 1. Delete large objects
        del current_training_params
        del current_ft_params
        del best_models_in_routine
        del routine

        # 2. Clear GPU and RAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Specifically for multi-process CUDA

    print(f"DONE: {routine_key} | Time: {time.time() - start_time:.2f}s")
    return routine_key


# ---------------------------------------------------------
# 4. MAIN EXECUTION (Parallel)
# ---------------------------------------------------------
if __name__ == "__main__":
    absolute_start_time = time.time()
    routine_keys = list(ROUTINE_REGISTERY.keys())

    print(f"Launching {len(routine_keys)} routines (Workers: {args.workers})")

    # max_nbytes=None and mmap_mode=None are default,
    # but we can use 'require="sharedmem"' if we had issues.
    # Here, we keep it simple.
    Parallel(n_jobs=args.workers)(delayed(execute_routine)(rk) for rk in routine_keys)

    print("\n" + "#" * 60)
    print(f"TOTAL EXECUTION TIME: {time.time() - absolute_start_time:.2f} seconds")
    print("#" * 60)
