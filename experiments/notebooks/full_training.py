import sys
import os

# 1. SETUP PATHS
project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import time
import gc
import numpy as np
import torch
import pandas as pd
from joblib import Parallel, delayed


from core.utils import (
    device_recognizer,
    datasets_loader,
    train_test_splitter,
    seed_everything,
    label_unifier,
    label_encoder,
    signal_target_decleration,
    load_best_params_lookup,
    p_subsampler_torch,
    Routine,
)
from core.routine_registery import ROUTINE_REGISTERY
from core.experiment_result import ExperimentResult
from easy_torchkit.src.configurations import TrainingParams
from experiments.notebooks.training_artifacts import (
    training_params_dict,
    fine_tuning_params_dict,
)

# ---------------------------------------------------------
# 2. ARGPARSE FOR TERMINAL CONTROL
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Full Experiment Runner")
parser.add_argument(
    "--subsample",
    type=float,
    default=None,
    help="Fine-tuning subsampling factor (e.g. 0.1)",
)
parser.add_argument(
    "--params_file",
    type=str,
    default="best_params.jsonl",
    help="Path to best_params.jsonl",
)
parser.add_argument(
    "--workers", type=int, default=3, help="Number of parallel routines"
)
parser.add_argument(
    "--reps",
    type=int,
    default=5,
    help="Number of repetitions for statistical significance",
)
args = parser.parse_args()

# Global settings from args
REPS = args.reps
FT_SUBSAMPLE = args.subsample
dataset_names = ["mfpt", "cwru", "kaist"]

# ---------------------------------------------------------
# 3. DATA PREPARATION (Shared across workers)
# ---------------------------------------------------------
device = device_recognizer()
os.makedirs("results/drawings", exist_ok=True)

raw_datasets = datasets_loader()
mfpt_data, cwru_data, kaist_data = label_unifier(
    raw_datasets["mfpt"], raw_datasets["cwru"], raw_datasets["kaist"]
)
x_mfpt, y_mfpt, x_cwru, y_cwru, x_kaist, y_kaist = signal_target_decleration(
    mfpt_data, cwru_data, kaist_data
)
y_mfpt_enc, y_cwru_enc, y_kaist_enc = label_encoder([y_mfpt, y_cwru, y_kaist])

experiment_data = {
    "mfpt": {"x": x_mfpt, "y": y_mfpt_enc},
    "cwru": {"x": x_cwru, "y": y_cwru_enc},
    "kaist": {"x": x_kaist, "y": y_kaist_enc},
}

# Load the hyperparameters found in the tuning phase
best_params_lookup = load_best_params_lookup(args.params_file)


# ---------------------------------------------------------
# 4. WORKER FUNCTION
# ---------------------------------------------------------
def run_routine_experiment(routine_key):
    import copy
    # Create a fresh copy of the routine config for this process
    routine_cfg = copy.deepcopy(ROUTINE_REGISTERY[routine_key])
    routine = Routine(**routine_cfg)

    if routine.description not in best_params_lookup:
        return f"Skipped {routine_key}: No params found in {args.params_file}."

    best_p_data = best_params_lookup[routine.description]
    res = ExperimentResult(reps=REPS, title=routine.description)

    # Initialize Containers
    for ds in dataset_names:
        res.results[ds], res.models[ds] = [], []
        res.fine_tuning_results[ds] = {tgt: [] for tgt in dataset_names if tgt != ds}
        res.fine_tuning_models[ds] = {tgt: [] for tgt in dataset_names if tgt != ds}

    try:
        # REPETITION LOOP
        for rep_idx, rnd_state in enumerate(res.random_states):
            seed_everything(rnd_state)
            splits = train_test_splitter(
                experiment_data, test_size=0.4, random_state=rnd_state
            )
            source_models_in_rep = {}

            # PHASE 1: SOURCE TRAINING
            for src_name in dataset_names:
                x_proc = routine.apply_preprocessing(splits[src_name]["train_x"], src_name)
                x_tensor = torch.tensor(x_proc, dtype=torch.float32).to(device)

                # Robust label conversion
                y_train_labels = splits[src_name]["train_y"]
                if hasattr(y_train_labels, "values"):
                    y_train_labels = y_train_labels.values
                y_tensor = torch.tensor(y_train_labels, dtype=torch.long).to(device)

                # Build Training Params
                best_p = best_p_data["training_best_params"][src_name]
                t_params = TrainingParams(
                    **{
                        **training_params_dict,
                        "lr": best_p["lr"],
                        "batch_size": best_p["batch_size"],
                    }
                )

                # --- REFINED MODEL ARG HANDLING ---
                # Get parameters from registry
                model_params = routine.model[1].get(src_name, {}).copy()
                
                # POP preprocessing args so they don't leak into ANY model constructor
                win_len_val = model_params.pop("win_len", None)
                model_params.pop("hop_len", None)

                # Setup standard kwargs
                model_kwargs = {
                    "device": device,
                    "random_state": rnd_state,
                    **model_params
                }
                
                model_type_name = routine.model[0].__name__

                # Logic for which models actually need an 'input_size' argument
                if model_type_name == "LSTMClassifier":
                    if win_len_val is not None:
                        model_kwargs["input_size"] = win_len_val
                
                elif model_type_name == "DNNClassifier":
                    # DNNs need the flattened feature dimension
                    model_kwargs["input_size"] = x_tensor.size()[-1]
                
                # NOTE: CNNClassifier is skipped here because it doesn't accept 'input_size'

                # 2. Model Init
                model = routine.model[0](**model_kwargs)
                model.fit(x_tensor, y_tensor, t_params)
                model.recover_best_model()

                source_models_in_rep[src_name] = model
                res.models[src_name].append(model)

                # Eval
                current_evals = {}
                for eval_name in dataset_names:
                    x_te = routine.apply_preprocessing(
                        splits[eval_name]["test_x"], eval_name, src_name
                    )
                    y_te_labels = splits[eval_name]["test_y"]
                    if hasattr(y_te_labels, "values"):
                        y_te_labels = y_te_labels.values
                    y_te_tensor = torch.tensor(y_te_labels, dtype=torch.long).to(device)
                    
                    current_evals[eval_name] = model.evaluate(
                        torch.tensor(x_te, dtype=torch.float32).to(device), y_te_tensor
                    )
                res.results[src_name].append(current_evals)

            # PHASE 2: FINE-TUNING
            for src_name in dataset_names:
                for tgt_name in [t for t in dataset_names if t != src_name]:
                    x_tgt = routine.apply_preprocessing(
                        splits[tgt_name]["train_x"], tgt_name, src_name
                    )
                    y_tgt_labels = splits[tgt_name]["train_y"]
                    if hasattr(y_tgt_labels, "values"):
                        y_tgt_labels = y_tgt_labels.values
                    y_tgt_tensor = torch.tensor(y_tgt_labels, dtype=torch.long).to(device)
                    x_tgt_tensor = torch.tensor(x_tgt, dtype=torch.float32).to(device)

                    if isinstance(FT_SUBSAMPLE, float):
                        x_tgt_tensor, y_tgt_tensor = p_subsampler_torch(
                            x_tgt_tensor, y_tgt_tensor, rnd_state, FT_SUBSAMPLE
                        )

                    best_ft_p = best_p_data["fine_tuning_best_params"][src_name][tgt_name]
                    ft_params = TrainingParams(
                        **{
                            **fine_tuning_params_dict,
                            "lr": best_ft_p["lr"],
                            "batch_size": best_ft_p["batch_size"],
                        }
                    )

                    # Fine-tuning a copy of the source model
                    ft_model = source_models_in_rep[src_name].copy(reset_history=False)
                    ft_model.fit(x_tgt_tensor, y_tgt_tensor, ft_params)
                    ft_model.recover_best_model()

                    res.fine_tuning_models[src_name][tgt_name].append(ft_model)

                    # Eval FT
                    ft_evals = {}
                    for eval_name in dataset_names:
                        x_te_ft = routine.apply_preprocessing(
                            splits[eval_name]["test_x"], eval_name, src_name
                        )
                        y_te_labels = splits[eval_name]["test_y"]
                        if hasattr(y_te_labels, "values"):
                            y_te_labels = y_te_labels.values
                        y_te_tensor = torch.tensor(y_te_labels, dtype=torch.long).to(device)
                        
                        ft_evals[eval_name] = ft_model.evaluate(
                            torch.tensor(x_te_ft, dtype=torch.float32).to(device),
                            y_te_tensor,
                        )
                    res.fine_tuning_results[src_name][tgt_name].append(ft_evals)

        # --- 5. EXPORT HISTORIES ---
        for ds in dataset_names:
            res.models[ds][-1].visualize_training_history(
                title=f"{res.title} - Source Training on {ds.upper()}",
                show_or_export="export",
                export_path=f"results/drawings/{res.title}_Source_{ds.upper()}",
            )
            for tgt in [t for t in dataset_names if t != ds]:
                res.fine_tuning_models[ds][tgt][-1].visualize_training_history(
                    title=f"FT: {ds.upper()} to {tgt.upper()} ({res.title})",
                    show_or_export="export",
                    export_path=f"results/drawings/FT_{ds.upper()}_to_{tgt.upper()}_{res.title}",
                )

        # --- 6. TIMING METRICS ---
        timing_summary = {
            "source": {
                ds: {
                    "total_time": np.mean([m.history[-1].total_time for m in m_list]),
                    "avg_epoch": np.mean([m.history[-1].average_epoch_time for m in m_list]),
                }
                for ds, m_list in res.models.items()
            },
            "fine_tuning": {
                src: {
                    tgt: {
                        "total_time": np.mean([m.history[-1].total_time for m in m_list]),
                        "avg_epoch": np.mean([m.history[-1].average_epoch_time for m in m_list]),
                    }
                    for tgt, m_list in tgt_dict.items()
                }
                for src, tgt_dict in res.fine_tuning_models.items()
            },
        }
        res.__dict__["timing_summary"] = timing_summary

        res.log_to_jsonl(
            log_file=f"results/{routine_key}.jsonl",
            exclude=["models", "fine_tuning_models"],
        )

    finally:
        # Cleanup routine resources
        del res, source_models_in_rep
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
    return f"Finished {routine_key}"


# ---------------------------------------------------------
# 7. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    routine_keys = list(ROUTINE_REGISTERY.keys())

    print(f"Starting {len(routine_keys)} experiments with {args.workers} workers...")
    
    # Use backend='loky' for better worker isolation
    Parallel(n_jobs=args.workers, backend="loky")(
        delayed(run_routine_experiment)(rk) for rk in routine_keys
    )

    print(f"\nALL EXPERIMENTS FINISHED. Total Time: {time.time() - start:.2f}s")