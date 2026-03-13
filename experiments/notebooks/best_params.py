### --- Setting the base directory ---
import sys
import os

project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)
import argparse
from typing import Literal
import time
from easy_torchkit.src.contracts.training_params import TrainingParams
import numpy as np
import torch
import gc  # Added for garbage collection
from joblib import Parallel, delayed


### --- Imports ---
from core.utils import (
    datasets_loader,
    label_unifier,
    load_best_params_lookup,
    signal_target_decleration,
    label_encoder,
    train_test_splitter,
    extract_study_data,
    torch_sampler,
    Routine,
    BestParams,
)
from core.routine_registery import ROUTINE_REGISTERY, fine_tuning_search_space
from core.best_params_finder import tune_source_phase, tune_fine_tuning_phase
from experiments.notebooks.training_artifacts import training_params_step_dict

# ---------------------------------------------------------
# 1. ARGPARSE FOR TERMINAL CONTROL
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning Routines")
parser.add_argument(
    "--phase",
    type=str,
    choices=["both", "fine_tuning"],
    default="both",
    help="Select the phase to find best hyperparameters.",
)

parser.add_argument(
    "--best_params_file",
    type=str,
    default=None,
    help="Select the dataset to use.",
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

PHASE = args.phase
BEST_PARAMS_FILE = args.best_params_file
SUBSAMPLING_STYLE = args.subsampling_style
SUBSAMPLING_FACTOR = args.subsampling_factor
FINE_TUNING_STYLE = args.fine_tuning_style
DEVICE = torch.device(args.device)
WORKERS = args.workers


# ---------------------------------------------------------
# 2. INITIAL SETTINGS (Shared across workers)
# ---------------------------------------------------------
p_suffix = (
    f"_ph_{str(PHASE).replace('.', '')}"
    f"_fs_{str(FINE_TUNING_STYLE).replace('.', '')}"
    f"_ss_{str(SUBSAMPLING_STYLE).replace('.', '')}"
    f"_sp_{str(SUBSAMPLING_FACTOR).replace('.', '')}"
)
output_dir = "results/best_params/"
os.makedirs(output_dir, exist_ok=True)

datasets = datasets_loader()
if BEST_PARAMS_FILE:
    best_params_lookup = load_best_params_lookup(BEST_PARAMS_FILE)
else:
    best_params_lookup = None

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
                DEVICE
            )
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)

            model_kwargs = routine.model[1].get(ds_name, {}).copy()

            if routine.model[0].__name__ == "LSTMClassifier":
                win_len_val = model_kwargs.pop("win_len", None)
                model_kwargs.pop("hop_len", None)
                if win_len_val is not None:
                    model_kwargs["input_size"] = win_len_val

            model_instance = routine.model[0](
                device=DEVICE, random_state=random_state, **model_kwargs
            )
            if PHASE == "both":
                params, model, training_study = tune_source_phase(
                    model_instance,
                    ds_name,
                    x_train_tensor,
                    y_train,
                    routine.training_best_params_ranges,
                    training_params_step_dict["training"],
                )
                current_training_params[ds_name] = params
                best_models_in_routine[ds_name] = model
            else:
                fixed_training_params = best_params_lookup[routine.description][
                    "training_best_params"
                ]

                src_hp = fixed_training_params[ds_name]
                t_params = TrainingParams(
                    **{
                        **training_params_step_dict["training"],
                        "lr": src_hp["lr"],
                        "batch_size": src_hp["batch_size"],
                    }
                )

                model_instance.fit(x_train_tensor, y_train, t_params)
                model_instance.recover_best_model()
                best_models_in_routine[ds_name] = model_instance

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
                ).to(DEVICE)

                x_target_tensor = torch.tensor(x_target, dtype=torch.float32).to(DEVICE)

                if SUBSAMPLING_FACTOR and SUBSAMPLING_STYLE:
                    x_target_tensor, y_target = torch_sampler(
                        x=x_target_tensor,
                        y=y_target,
                        rnd_state=random_state,
                        subsampling_style=SUBSAMPLING_STYLE,
                        subsampling_factor=SUBSAMPLING_FACTOR,
                    )

                if FINE_TUNING_STYLE == "supervised":
                    ft_params, _, fine_tuning_study = tune_fine_tuning_phase(
                        best_models_in_routine[src_name],
                        tgt_name,
                        x_target_tensor,
                        y_target,
                        routine.training_best_params_ranges,
                        training_params_step_dict["fine_tuning"]["supervised"],
                    )

                elif FINE_TUNING_STYLE == "contrastive":
                    ft_params, _, fine_tuning_study = tune_fine_tuning_phase(
                        best_models_in_routine[src_name],
                        tgt_name,
                        x_target_tensor,
                        y_target,
                        routine.training_best_params_ranges,
                        training_params_step_dict["fine_tuning"]["contrastive"],
                        fine_tuning_style_search_space=fine_tuning_search_space[
                            "contrastive"
                        ],
                    )

                elif FINE_TUNING_STYLE == "dynamic_bootstrapping":
                    ft_params, _, fine_tuning_study = tune_fine_tuning_phase(
                        best_models_in_routine[src_name],
                        tgt_name,
                        x_target_tensor,
                        y_target,
                        routine.training_best_params_ranges,
                        training_params_step_dict["fine_tuning"][
                            "dynamic_bootstrapping"
                        ],
                        fine_tuning_style_search_space=fine_tuning_search_space[
                            "dynamic_bootstrapping"
                        ],
                    )

                else:
                    raise ValueError(
                        f"Unsupported fine-tuning style: {FINE_TUNING_STYLE}"
                    )

                current_ft_params[src_name][tgt_name] = ft_params

        if PHASE == "both":
            training_study_details = extract_study_data(training_study)
            training_best_params = current_training_params
        elif PHASE == "fine_tuning":
            training_study_details = None
            training_best_params = fixed_training_params

        # Export Results
        bp = BestParams(
            description=routine.description,
            training_best_params=training_best_params,
            fine_tuning_best_params=current_ft_params,
            random_state=random_state,
            training_study_details=training_study_details,
            fine_tuning_study_details=extract_study_data(fine_tuning_study),
        )
        p_suffix = (
            f"_fs_{str(FINE_TUNING_STYLE)}"
            f"_ss_{str(SUBSAMPLING_STYLE)}"
            f"_sp_{str(SUBSAMPLING_FACTOR).replace('.', '')}"
        )
        bp.log_to_jsonl(
            log_file=f"{output_dir}best_params{p_suffix}.jsonl",
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

    print(f"Launching {len(routine_keys)} routines (Workers: {WORKERS})")

    # max_nbytes=None and mmap_mode=None are default,
    # but we can use 'require="sharedmem"' if we had issues.
    # Here, we keep it simple.
    Parallel(n_jobs=WORKERS)(delayed(execute_routine)(rk) for rk in routine_keys)

    print("\n" + "#" * 60)
    print(f"TOTAL EXECUTION TIME: {time.time() - absolute_start_time:.2f} seconds")
    print("#" * 60)
