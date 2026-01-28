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
import json
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
# 1. ARGPARSE & DYNAMIC PATHING
# ---------------------------------------------------------
default_device = device_recognizer()

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
    help="Path to the best_params.jsonl file",
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
parser.add_argument(
    "--device",
    type=str,
    default=str(default_device),
    help="Device to use (e.g., 'cuda:0', 'cpu')",
)
args = parser.parse_args()

# Dynamic naming: e.g., results_p_01 or results_p_10 for None
p_val_str = str(args.subsample).replace('.', '') if args.subsample is not None else "10"
output_dir = f"results_p_{p_val_str}"
drawings_dir = os.path.join(output_dir, "drawings")
os.makedirs(drawings_dir, exist_ok=True)

# Global Settings
DEVICE = torch.device(args.device)
REPS = args.reps
FT_SUBSAMPLE = args.subsample
dataset_names = ["mfpt", "cwru", "kaist"]

# ---------------------------------------------------------
# 2. DATA PREPARATION
# ---------------------------------------------------------
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

best_params_lookup = load_best_params_lookup(args.params_file)

# ---------------------------------------------------------
# 3. WORKER FUNCTION
# ---------------------------------------------------------
def run_routine_experiment(routine_key):
    import copy
    routine_cfg = copy.deepcopy(ROUTINE_REGISTERY[routine_key])
    routine = Routine(**routine_cfg)

    if routine.description not in best_params_lookup:
        return None

    best_p_data = best_params_lookup[routine.description]
    res = ExperimentResult(reps=REPS, title=routine.description)

    # Init Containers
    for ds in dataset_names:
        res.results[ds], res.models[ds] = [], []
        res.fine_tuning_results[ds] = {tgt: [] for tgt in dataset_names if tgt != ds}
        res.fine_tuning_models[ds] = {tgt: [] for tgt in dataset_names if tgt != ds}

    try:
        for rep_idx, rnd_state in enumerate(res.random_states):
            seed_everything(rnd_state)
            splits = train_test_splitter(experiment_data, test_size=0.4, random_state=rnd_state)
            source_models_in_rep = {}

            # --- PHASE 1: SOURCE TRAINING ---
            for src_name in dataset_names:
                x_proc = routine.apply_preprocessing(splits[src_name]["train_x"], src_name)
                x_tensor = torch.tensor(x_proc, dtype=torch.float32).to(DEVICE)
                
                y_labels = splits[src_name]["train_y"]
                y_tensor = torch.tensor(y_labels.values if hasattr(y_labels, "values") else y_labels, 
                                        dtype=torch.long).to(DEVICE)

                best_p = best_p_data["training_best_params"][src_name]
                t_params = TrainingParams(**{**training_params_dict, "lr": best_p["lr"], "batch_size": best_p["batch_size"]})

                model_params = routine.model[1].get(src_name, {}).copy()
                win_len_val = model_params.pop("win_len", None)
                model_params.pop("hop_len", None)

                model_kwargs = {"device": DEVICE, "random_state": rnd_state, **model_params}
                model_type_name = routine.model[0].__name__
                if model_type_name == "LSTMClassifier" and win_len_val:
                    model_kwargs["input_size"] = win_len_val
                elif model_type_name == "DNNClassifier":
                    model_kwargs["input_size"] = x_tensor.size()[-1]

                model = routine.model[0](**model_kwargs)
                model.fit(x_tensor, y_tensor, t_params)
                model.recover_best_model()

                source_models_in_rep[src_name] = model
                res.models[src_name].append(model)

                # Eval Source
                curr_evals = {}
                for eval_name in dataset_names:
                    x_te = routine.apply_preprocessing(splits[eval_name]["test_x"], eval_name, src_name)
                    y_te_raw = splits[eval_name]["test_y"]
                    y_te_ten = torch.tensor(y_te_raw.values if hasattr(y_te_raw, "values") else y_te_raw, 
                                            dtype=torch.long).to(DEVICE)
                    curr_evals[eval_name] = model.evaluate(torch.tensor(x_te, dtype=torch.float32).to(DEVICE), y_te_ten)
                res.results[src_name].append(curr_evals)

            # --- PHASE 2: FINE-TUNING ---
            for src_name in dataset_names:
                for tgt_name in [t for t in dataset_names if t != src_name]:
                    x_tgt = routine.apply_preprocessing(splits[tgt_name]["train_x"], tgt_name, src_name)
                    y_tgt_raw = splits[tgt_name]["train_y"]
                    y_tgt_ten = torch.tensor(y_tgt_raw.values if hasattr(y_tgt_raw, "values") else y_tgt_raw, 
                                             dtype=torch.long).to(DEVICE)
                    x_tgt_ten = torch.tensor(x_tgt, dtype=torch.float32).to(DEVICE)

                    if FT_SUBSAMPLE:
                        x_tgt_ten, y_tgt_ten = p_subsampler_torch(x_tgt_ten, y_tgt_ten, rnd_state, FT_SUBSAMPLE)

                    best_ft_p = best_p_data["fine_tuning_best_params"][src_name][tgt_name]
                    ft_params = TrainingParams(**{**fine_tuning_params_dict, "lr": best_ft_p["lr"], "batch_size": best_ft_p["batch_size"]})

                    ft_model = source_models_in_rep[src_name].copy(reset_history=False)
                    ft_model.fit(x_tgt_ten, y_tgt_ten, ft_params)
                    ft_model.recover_best_model()
                    res.fine_tuning_models[src_name][tgt_name].append(ft_model)

                    # Eval FT
                    ft_evals = {}
                    for eval_name in dataset_names:
                        x_te_ft = routine.apply_preprocessing(splits[eval_name]["test_x"], eval_name, src_name)
                        y_te_raw = splits[eval_name]["test_y"]
                        y_te_ten = torch.tensor(y_te_raw.values if hasattr(y_te_raw, "values") else y_te_raw, 
                                                dtype=torch.long).to(DEVICE)
                        ft_evals[eval_name] = ft_model.evaluate(torch.tensor(x_te_ft, dtype=torch.float32).to(DEVICE), y_te_ten)
                    res.fine_tuning_results[src_name][tgt_name].append(ft_evals)

        # --- EXPORT FT HISTORIES ONLY ---
        for ds in dataset_names:
            for tgt in [t for t in dataset_names if t != ds]:
                export_fn = f"FT_{ds.upper()}_to_{tgt.upper()}_{res.title}_p_{p_val_str}"
                res.fine_tuning_models[ds][tgt][-1].visualize_training_history(
                    title=f"FT: {ds.upper()} to {tgt.upper()} ({res.title}) Sub:{FT_SUBSAMPLE}",
                    show_or_export="export",
                    export_path=os.path.join(drawings_dir, export_fn),
                )

        # Build timing summary
        timing_summary = {
            "source": {ds: {"total_time": np.mean([m.history[-1].total_time for m in m_list])} for ds, m_list in res.models.items()},
            "fine_tuning": {src: {tgt: {"total_time": np.mean([m.history[-1].total_time for m in m_list])} for tgt, m_list in tgt_dict.items()} for src, tgt_dict in res.fine_tuning_models.items()}
        }

        # Return a manually constructed dictionary to avoid 'to_dict' issues
        return {
            "title": res.title,
            "random_states": [int(s) for s in res.random_states],
            "results": res.results,
            "fine_tuning_results": res.fine_tuning_results,
            "timing_summary": timing_summary,
            "subsample_factor": FT_SUBSAMPLE
        }

    finally:
        del res, source_models_in_rep
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# ---------------------------------------------------------
# 4. EXECUTION & MASTER LOGGING
# ---------------------------------------------------------
if __name__ == "__main__":
    start_all = time.time()
    routine_keys = list(ROUTINE_REGISTERY.keys())

    print(f"Launching Experiments | Subsample: {FT_SUBSAMPLE} | Workers: {args.workers} | Device: {DEVICE}")

    # Parallel execution
    results_list = Parallel(n_jobs=args.workers, backend="loky")(
        delayed(run_routine_experiment)(rk) for rk in routine_keys
    )

    # Filter skips and write master file
    results_list = [r for r in results_list if r is not None]
    master_log_path = os.path.join(output_dir, f"full_results_p_{p_val_str}.jsonl")
    
    with open(master_log_path, "w") as f:
        for entry in results_list:
            f.write(json.dumps(entry) + "\n")

    print(f"\nCOMPLETED. Results in {output_dir}. Total Time: {time.time() - start_all:.2f}s")