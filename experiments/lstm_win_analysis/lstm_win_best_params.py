import sys
import os
import argparse
import time
import numpy as np
import torch
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Any, Optional, Callable, Type, Literal
import copy
from pathlib import Path

# --- Setting the base directory ---
project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

### --- Core Imports ---
from core.utils import (
    device_recognizer,
    datasets_loader,
    label_unifier,
    signal_target_decleration,
    label_encoder,
    train_test_splitter,
    extract_study_data,
    Routine,
    BestParams,
)
from core.routine_registery import ROUTINE_REGISTERY
from core.best_params_finder import tune_source_phase
from experiments.notebooks.training_artifacts import training_params_dict

# ---------------------------------------------------------
# 1. INITIAL SETTINGS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Find Best Window Length and Training Params with Visualization"
)
parser.add_argument(
    "--workers", type=int, default=3, help="Number of routines in parallel."
)
args = parser.parse_args()

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
# 2. THE WORKER FUNCTION
# ---------------------------------------------------------
def execute_routine(routine_key):
    """
    Finds and logs best params/loss for EVERY dataset-window combination using NESTED keys.
    """
    print(f"\n>>> Starting Routine: {routine_key}")

    safe_routine_name = routine_key.replace("->", "_")
    output_dir = Path(f"experiment_results/{safe_routine_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initializing the nested storage
    # Structure: { "mfpt": { "win_10": {...} }, "cwru": { ... } }
    nested_results = {}

    window_search_space = [10, 20, 100, 200, 400, 800, 1200, 2400]
    random_state = np.random.randint(1, 1000)

    base_routine_dict = ROUTINE_REGISTERY[routine_key]
    routine = Routine(**copy.deepcopy(base_routine_dict))

    raw_splits = train_test_splitter(
        experiment_data, test_size=0.4, random_state=random_state
    )

    start_time = time.time()

    for ds_name in ["mfpt", "cwru", "kaist"]:
        print(f"  [Dataset: {ds_name}] Iterating window lengths...")

        # Initialize the sub-dictionary for the dataset
        nested_results[ds_name] = {}

        for w_len in window_search_space:
            # A. Update routine config
            if w_len is not None:
                for func, all_kwargs in routine.data_processing:
                    if func.__name__ == "sequence_data":
                        if ds_name not in all_kwargs:
                            all_kwargs[ds_name] = {}
                        all_kwargs[ds_name]["win_len"] = w_len
                        all_kwargs[ds_name]["hop_len"] = w_len

            # B. Preprocess
            x_train = routine.apply_preprocessing(
                raw_splits[ds_name]["train_x"], ds_name
            )
            y_train = torch.tensor(raw_splits[ds_name]["train_y"], dtype=torch.long).to(
                device
            )
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)

            # C. Model init
            model_kwargs = {"input_size": w_len}
            model_instance = routine.model[0](
                device=device, random_state=random_state, **model_kwargs
            )

            # D. Run Optuna Tuning
            study_params, model, study = tune_source_phase(
                model=model_instance,
                dataset_name=ds_name,
                train_x=x_train_tensor,
                train_y=y_train,
                search_space=routine.training_best_params_ranges,
                training_params_dict=training_params_dict,
            )

            # E. Visualize
            viz_title = f"DS-{ds_name}_Win{w_len}_{safe_routine_name}"
            viz_path = output_dir / f"{viz_title}.png"
            model.visualize_training_history(
                title=viz_title, show_or_export="export", export_path=str(viz_path)
            )

            # --- F. NESTED LOGGING ---
            result_entry = copy.deepcopy(study_params)
            result_entry["window_len"] = w_len
            result_entry["best_val_loss"] = study.best_value

            # Store in nested format
            nested_results[ds_name][f"win_{w_len}"] = result_entry

    # --- Export Metadata ---
    bp = BestParams(
        description=routine.description,
        training_best_params=nested_results,
        fine_tuning_best_params={},
        random_state=random_state,
        training_study_details=None,  # Removed as per requirement
        fine_tuning_study_details=None,
    )

    # Save a unique log file per routine
    log_filename = f"best_params_{safe_routine_name}.jsonl"
    bp.log_to_jsonl(log_file=log_filename)

    print(f"DONE: {routine_key} | Total Time: {time.time() - start_time:.2f}s")
    return routine_key


# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    absolute_start_time = time.time()
    routine_keys = [
        "raw->scaled->sequenced->lstm",
        "raw->env->scaled->sequenced->lstm",
    ]

    print(f"Launching {len(routine_keys)} routines in parallel...")
    Parallel(n_jobs=args.workers)(delayed(execute_routine)(rk) for rk in routine_keys)

    print("\n" + "#" * 60)
    print(f"TOTAL EXECUTION TIME: {time.time() - absolute_start_time:.2f} seconds")
    print("#" * 60)
