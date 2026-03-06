import sys
import os
import argparse
import time
import numpy as np
import torch
from joblib import Parallel, delayed
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
    Routine,
    BestParams,
    sequence_data,
)
from damavand.damavand.signal_processing.transformations import env
from damavand.damavand.utils import z_score_scaler
from core.model_repository import LSTMClassifier
from core.best_params_finder import tune_source_phase
from experiments.notebooks.training_artifacts import training_params_dict

# ---------------------------------------------------------
# 1. LOCAL DEFINITIONS (MATCHING YOUR REGISTRY)
# ---------------------------------------------------------
# Exact description strings from your registry
DESC_RAW = "Raw -> Scaled -> Sequenced -> LSTM"
DESC_ENV = "Raw -> Env -> Scaled -> Sequenced -> LSTM"

# Exact Search Spaces from your registry
lstm_best_params_ranges = {
    "mfpt": {
        "lr": [0.0005, 0.0001, 0.00005, 0.00001],
        "batch_size": [8, 16, 32, 64, 128],
    },
    "cwru": {
        "lr": [0.0005, 0.0001, 0.00005, 0.00001],
        "batch_size": [8, 16, 32, 64, 128],
    },
    "kaist": {
        "lr": [0.0005, 0.0001, 0.00005, 0.00001],
        "batch_size": [8, 16, 32, 64, 128],
    },
}

# ---------------------------------------------------------
# 2. INITIAL SETTINGS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Find Best Window Length")
parser.add_argument("--workers", type=int, default=2)
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
# 3. THE WORKER FUNCTION
# ---------------------------------------------------------
def execute_routine(routine_key):
    start_time = time.time()

    # Map the key to your EXACT description string from the registry
    if routine_key == "raw->scaled->sequenced->lstm":
        description = "Raw -> Scaled -> Sequenced -> LSTM"
        use_env = False
    else:
        description = "Raw -> Env -> Scaled -> Sequenced -> LSTM"
        use_env = True

    print(f"\n>>> Starting Routine: {description}")

    safe_folder_name = routine_key.replace("->", "_")
    output_dir = Path(f"experiment_results/{safe_folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    nested_results = {}
    window_search_space = [10, 20, 100, 200, 400, 800, 1200, 2400]
    random_state = np.random.randint(1, 1000)

    raw_splits = train_test_splitter(
        experiment_data, test_size=0.4, random_state=random_state
    )

    for ds_name in ["mfpt", "cwru", "kaist"]:
        print(f"  [Dataset: {ds_name}] Iterating window lengths...")
        nested_results[ds_name] = {}

        for w_len in window_search_space:
            # --- STEP A: BUILD THE PROCESSING PIPELINE TO MATCH YOUR ROUTINE CLASS ---
            processing_steps = []

            if use_env:
                processing_steps.append((env, {"mfpt": {}, "cwru": {}, "kaist": {}}))

            processing_steps.append(
                (
                    z_score_scaler,
                    {
                        "mfpt": {"axis": 1, "return_df": False},
                        "cwru": {"axis": 1, "return_df": False},
                        "kaist": {"axis": 1, "return_df": False},
                    },
                )
            )

            # CRITICAL FIX: Nest the dictionary to satisfy:
            # all_kwargs.get(self.description).get(lookup_key)
            processing_steps.append(
                (
                    sequence_data,
                    {description: {ds_name: {"win_len": w_len, "hop_len": w_len}}},
                )
            )

            # --- STEP B: INITIALIZE ROUTINE ---
            iter_routine = Routine(
                description=description,
                data_processing=processing_steps,
                model=(LSTMClassifier, {}),
                training_best_params_ranges=lstm_best_params_ranges,
            )

            # --- STEP C: PREPROCESS ---
            try:
                # Now apply_preprocessing will find the w_len under the description key!
                x_train = iter_routine.apply_preprocessing(
                    raw_splits[ds_name]["train_x"], ds_name
                )

                actual_win = x_train.shape[-1]
                if actual_win != w_len:
                    print(
                        f"    !! SHAPE MISMATCH: Expected {w_len}, got {actual_win}. Skipping..."
                    )
                    continue

                x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
                y_train = torch.tensor(
                    raw_splits[ds_name]["train_y"], dtype=torch.long
                ).to(device)

                # --- STEP D: MODEL & TUNING ---
                model_instance = LSTMClassifier(
                    device=device, random_state=random_state, input_size=w_len
                )

                # DEBUG: Print shape once to be sure
                if w_len == window_search_space[0]:
                    print(
                        f"DEBUG: Data shapes -> x_train: {x_train_tensor.shape}, y_train: {y_train.shape}"
                    )

                study_params, trained_model, study = tune_source_phase(
                    model=model_instance,
                    dataset_name=ds_name,
                    train_x=x_train_tensor,
                    train_y=y_train,
                    search_space=lstm_best_params_ranges,
                    training_params_dict=training_params_dict,
                )

                # Save history and results as before...
                viz_title = f"DS-{ds_name}_Win{w_len}"
                trained_model.visualize_training_history(
                    title=viz_title,
                    show_or_export="export",
                    export_path=str(output_dir / f"{viz_title}.png"),
                )

                res = copy.deepcopy(study_params)
                res["window_len"] = w_len
                res["best_val_loss"] = study.best_value
                nested_results[ds_name][f"win_{w_len}"] = res

            except Exception as e:
                print(f"    !! Error on {ds_name} at win {w_len}: {e}")
                continue

    # Export results to jsonl as before...
    bp = BestParams(
        description=description,
        training_best_params=nested_results,
        random_state=random_state,
    )
    bp.log_to_jsonl(log_file=f"best_params_{safe_folder_name}.jsonl")
    return routine_key


# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    routine_keys = ["raw->scaled->sequenced->lstm", "raw->env->scaled->sequenced->lstm"]

    print(f"Launching Analysis for {len(routine_keys)} routines...")
    Parallel(n_jobs=args.workers)(delayed(execute_routine)(rk) for rk in routine_keys)

    print(f"\nTOTAL EXECUTION TIME: {time.time() - t0:.2f} seconds")
