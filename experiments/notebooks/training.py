### --- Setting the base directory ---
import sys
import os
import time

project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

from core.experiment_result import ExperimentResult
from core.routine_registery import ROUTINE_REGISTERY
from core.utils import (
    Routine,
    datasets_loader,
    label_encoder,
    label_unifier,
    load_best_params_lookup,
    seed_everything,
    signal_target_decleration,
    torch_sampler,
    train_test_splitter,
)
from easy_torchkit.src.contracts.training_params import TrainingParams
from experiments.notebooks.training_artifacts import training_params_step_dict
import argparse
import torch
from joblib import Parallel, delayed
import numpy as np
import gc

# ---------------------------------------------------------
# 1. ARGPARSE FOR TERMINAL CONTROL
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Experiments using the best hyperparameters, repetitively"
)

parser.add_argument(
    "--best_params_file",
    type=str,
    help="Select the dataset to use.",
)

parser.add_argument(
    "--reps",
    type=int,
    default=5,
    help="Number of repetitions for statistical significance",
)

parser.add_argument(
    "--subsampling_style",
    type=str,
    choices=["percentage", "shots_per_class", None],
    default=None,
    help="How the fine-tuning set should be subsampled. Default is None, meaning no subsampling.",
)

parser.add_argument(
    "--subsampling_factor",
    type=float,
    default=None,
    help="Subsampling factor for fine-tuning; if the style is percentage, this is the percentage (e.g., 0.1 for 10%%)."
    "If the style is shots_per_class, this is the number of shots per class. Default is None which matches the subsampling_style of None;"
    "this means no subsampling is performed.",
)

parser.add_argument(
    "--fine_tuning_style",
    type=str,
    choices=["supervised", "contrastive", "dynamic_bootstrapping"],
    default="supervised",
    help="The fine-tuning style to use. Default is 'supervised'.",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="The device to use for training. Default is 'cuda:0'.",
)

parser.add_argument(
    "--workers",
    type=int,
    default=3,
    help="Number of routines to run in parallel. Default is 3.",
)
args = parser.parse_args()

REPS = args.reps
BEST_PARAMS_FILE = args.best_params_file
SUBSAMPLING_STYLE = args.subsampling_style
SUBSAMPLING_FACTOR = args.subsampling_factor
FINE_TUNING_STYLE = args.fine_tuning_style
DEVICE = torch.device(args.device)
WORKERS = args.workers

dataset_names = ["mfpt", "cwru", "kaist"]

p_suffix = (
    f"_fs_{str(FINE_TUNING_STYLE)}"
    f"_ss_{str(SUBSAMPLING_STYLE)}"
    f"_sp_{str(SUBSAMPLING_FACTOR).replace('.', '')}"
)
output_dir = f"results/results{p_suffix}/"
print(output_dir)
os.makedirs(output_dir, exist_ok=True)
drawings_dir = os.path.join(output_dir, "drawings")
os.makedirs(drawings_dir, exist_ok=True)


# ---------------------------------------------------------
# 2. DATA PREPARATION (Shared across workers)
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

best_params_lookup = load_best_params_lookup(BEST_PARAMS_FILE)


def run_routine_experiment(routine_key):
    import copy

    routine_cfg = copy.deepcopy(ROUTINE_REGISTERY[routine_key])
    routine = Routine(**routine_cfg)

    if routine.description not in best_params_lookup:
        return None

    best_params = best_params_lookup[routine.description]
    res = ExperimentResult(
        reps=REPS,
        title=routine.description,
        fine_tuning_style=FINE_TUNING_STYLE,
        subsampling_style=SUBSAMPLING_STYLE,
        subsampling_factor=SUBSAMPLING_FACTOR,
        best_params=best_params,
    )

    for ds in dataset_names:
        res.results[ds], res.models[ds], res.terminations[ds] = [], [], []
        res.fine_tuning_results[ds] = {tgt: [] for tgt in dataset_names if tgt != ds}
        res.fine_tuning_models[ds] = {tgt: [] for tgt in dataset_names if tgt != ds}
        res.fine_tuning_terminations[ds] = {
            tgt: [] for tgt in dataset_names if tgt != ds
        }

    try:
        for rep_idx, rnd_state in enumerate(res.random_states):
            seed_everything(rnd_state)
            splits = train_test_splitter(
                experiment_data, test_size=0.4, random_state=rnd_state
            )
            source_models_in_rep = {}

            # --- PHASE 1: SOURCE TRAINING ---
            for src_name in dataset_names:
                x_proc = routine.apply_preprocessing(
                    splits[src_name]["train_x"], src_name
                )
                x_tensor = torch.tensor(x_proc, dtype=torch.float32).to(DEVICE)

                y_labels = splits[src_name]["train_y"]
                y_tensor = torch.tensor(
                    y_labels.values if hasattr(y_labels, "values") else y_labels,
                    dtype=torch.long,
                ).to(DEVICE)

                best_params_dataset = best_params["training_best_params"][src_name]
                t_params = TrainingParams(
                    **{
                        **training_params_step_dict["training"],
                        "lr": best_params_dataset["lr"],
                        "batch_size": best_params_dataset["batch_size"],
                    }
                )

                model_params = routine.model[1].get(src_name, {}).copy()
                win_len_val = model_params.pop("win_len", None)
                model_params.pop("hop_len", None)

                model_kwargs = {
                    "device": DEVICE,
                    "random_state": rnd_state,
                    **model_params,
                }
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
                res.terminations[src_name].append(model.history[-1].termination)
                res.source_timing_raw[src_name].append(
                    {
                        "total_time": model.history[-1].total_time,
                        "average_epoch_time": model.history[-1].average_epoch_time,
                    }
                )

                # Eval Source
                curr_evals = {}
                for eval_name in dataset_names:
                    x_te = routine.apply_preprocessing(
                        splits[eval_name]["test_x"], eval_name, src_name
                    )
                    y_te_raw = splits[eval_name]["test_y"]
                    y_te_ten = torch.tensor(
                        y_te_raw.values if hasattr(y_te_raw, "values") else y_te_raw,
                        dtype=torch.long,
                    ).to(DEVICE)

                    curr_evals[eval_name] = model.evaluate(
                        torch.tensor(x_te, dtype=torch.float32).to(DEVICE), y_te_ten
                    )

                res.results[src_name].append(curr_evals)

            # --- PHASE 2: FINE TUNING ---
            for src_name in dataset_names:
                for tgt_name in [t for t in dataset_names if t != src_name]:
                    x_tgt = routine.apply_preprocessing(
                        splits[tgt_name]["train_x"], tgt_name, src_name
                    )
                    y_tgt_raw = splits[tgt_name]["train_y"]
                    y_tgt_ten = torch.tensor(
                        y_tgt_raw.values if hasattr(y_tgt_raw, "values") else y_tgt_raw,
                        dtype=torch.long,
                    ).to(DEVICE)
                    x_tgt_ten = torch.tensor(x_tgt, dtype=torch.float32).to(DEVICE)

                    if SUBSAMPLING_FACTOR and SUBSAMPLING_STYLE:
                        x_tgt_ten, y_tgt_ten = torch_sampler(
                            x=x_tgt_ten,
                            y=y_tgt_ten,
                            rnd_state=rnd_state,
                            subsampling_style=SUBSAMPLING_STYLE,
                            subsampling_factor=SUBSAMPLING_FACTOR,
                        )

                    best_ft_p = best_params["fine_tuning_best_params"][src_name][
                        tgt_name
                    ]
                    ft_params = TrainingParams(
                        **{
                            **training_params_step_dict["fine_tuning"][
                                FINE_TUNING_STYLE
                            ],
                            "lr": best_ft_p["lr"],
                            "batch_size": best_ft_p["batch_size"],
                        }
                    )

                    if FINE_TUNING_STYLE == "supervised":
                        ft_model = source_models_in_rep[src_name].copy(
                            reset_history=False
                        )
                        ft_model.fit(x_tgt_ten, y_tgt_ten, ft_params)
                        ft_model.recover_best_model()
                        res.fine_tuning_models[src_name][tgt_name].append(ft_model)

                    res.fine_tuning_timing_raw[src_name][tgt_name].append(
                        {
                            "total_time": ft_model.history[-1].total_time,
                            "average_epoch_time": ft_model.history[
                                -1
                            ].average_epoch_time,
                        }
                    )
                    res.fine_tuning_terminations[src_name][tgt_name].append(
                        model.history[-1].termination
                    )

                    # Eval FT
                    ft_evals = {}
                    for eval_name in dataset_names:
                        x_te_ft = routine.apply_preprocessing(
                            splits[eval_name]["test_x"], eval_name, src_name
                        )
                        y_te_raw = splits[eval_name]["test_y"]
                        y_te_ten = torch.tensor(
                            y_te_raw.values
                            if hasattr(y_te_raw, "values")
                            else y_te_raw,
                            dtype=torch.long,
                        ).to(DEVICE)
                        ft_evals[eval_name] = ft_model.evaluate(
                            torch.tensor(x_te_ft, dtype=torch.float32).to(DEVICE),
                            y_te_ten,
                        )
                    res.fine_tuning_results[src_name][tgt_name].append(ft_evals)

        for ds in dataset_names:
            if (SUBSAMPLING_FACTOR and SUBSAMPLING_STYLE) is None:
                res.models[ds][-1].visualize_training_history(
                    title=f"{res.title} - Source Training on {ds.upper()}",
                    show_or_export="export",
                    export_path=f"{drawings_dir}/{res.title}_Source_{ds.upper()}",
                )
            for tgt in [t for t in dataset_names if t != ds]:
                res.fine_tuning_models[ds][tgt][-1].visualize_training_history(
                    title=f"{res.title} -Source from {ds.upper()} Fine Tuning on {tgt.upper()} - {p_suffix}",
                    show_or_export="export",
                    export_path=f"{drawings_dir}/{res.title}_FineTuning_{tgt.upper()}",
                )

        res.log_to_jsonl(
            log_file=f"{output_dir}{p_suffix}.jsonl",
            exclude=["models", "fine_tuning_models"],
        )

    finally:
        del res, source_models_in_rep
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return f"Finished {routine_key}."


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
