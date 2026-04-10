import os
import sys

project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

print(project_root)

import re
from pathlib import Path

from core.result_processor import process_jsonl_to_dfs, export_to_xlsx

if __name__ == "__main__":
    files = [
        # "results/results_fs_supervised_ss_None_sp_None/_fs_supervised_ss_None_sp_None.jsonl",
        # "results/results_fs_supervised_ss_percentage_sp_01/_fs_supervised_ss_percentage_sp_01.jsonl",
        # "results/results_fs_supervised_ss_percentage_sp_005/_fs_supervised_ss_percentage_sp_005.jsonl",
        # "results/results_fs_supervised_ss_percentage_sp_001/_fs_supervised_ss_percentage_sp_001.jsonl",
        # "results/results_fs_supervised_ss_shots_per_class_sp_50/_fs_supervised_ss_shots_per_class_sp_50.jsonl",
        # "results/results_fs_supervised_ss_shots_per_class_sp_100/_fs_supervised_ss_shots_per_class_sp_100.jsonl",
        # "results/results_fs_supervised_ss_shots_per_class_sp_150/_fs_supervised_ss_shots_per_class_sp_150.jsonl",
        # "results/results_fs_supervised_ss_shots_per_class_sp_200/_fs_supervised_ss_shots_per_class_sp_200.jsonl",
        # "results/results_fs_supervised_ss_shots_per_class_sp_250/_fs_supervised_ss_shots_per_class_sp_250.jsonl",
        "results/results_fs_dynamic_bootstrapping_ss_None_sp_None/_fs_dynamic_bootstrapping_ss_None_sp_None.jsonl",
        "results/results_fs_dynamic_bootstrapping_gt_recovery_075_ss_None_sp_None/_fs_dynamic_bootstrapping_gt_recovery_075_ss_None_sp_None.jsonl",
        "results/results_fs_dynamic_bootstrapping_gt_recovery_05_ss_None_sp_None/_fs_dynamic_bootstrapping_gt_recovery_05_ss_None_sp_None.jsonl",
        "results/results_fs_dynamic_bootstrapping_gt_recovery_025_ss_None_sp_None/_fs_dynamic_bootstrapping_gt_recovery_025_ss_None_sp_None.jsonl",
    ]
    output_dir = "results/summarized/"
    os.makedirs(output_dir, exist_ok=True)

    routine_layout = {
        (0, 0): "Raw -> Scaled -> CNN",
        (0, 1): "Raw -> Env -> Scaled -> CNN",
        (1, 0): "Raw -> FFT -> Scaled -> CNN",
        (1, 1): "Raw -> Env -> FFT -> Scaled -> CNN",
        (2, 0): "Raw -> Zoomed FFT -> Scaled -> CNN",
        (2, 1): "Raw -> Env -> Zoomed FFT -> Scaled -> CNN",
        (3, 0): "Raw -> FFT -> Scaled -> Resampled -> DNN",
        (3, 1): "Raw -> Env -> FFT -> Scaled -> Resampled -> DNN",
        (4, 0): "Raw -> Zoomed FFT -> Scaled -> DNN",
        (4, 1): "Raw -> Env -> Zoomed FFT -> Scaled -> DNN",
        (5, 0): "Raw -> Scaled -> Sequenced -> LSTM",
        (5, 1): "Raw -> Env -> Scaled -> Sequenced -> LSTM",
    }

    for f_path in files:
        path = Path(f_path)
        if not path.exists():
            continue
        print(f"Processing {path.name}...")
        p_match = re.search(r"p_\d+", path.name)
        p_suffix = f"_{p_match.group(0)}" if p_match else ""
        data_dfs = process_jsonl_to_dfs(path)
        export_to_xlsx(
            data_dfs,
            routine_layout,
            os.path.join(
                output_dir, f"summarized{path.name.split('/')[-1].split('.')[0]}.xlsx"
            ),
            path.name,
        )

    print("\nProcessing complete.")
