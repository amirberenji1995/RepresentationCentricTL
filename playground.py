import pandas as pd
import json
import os
import re
from pathlib import Path


def process_jsonl_to_dfs(jsonl_path):
    routines_dict = {}
    datasets = ["mfpt", "cwru", "kaist"]

    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = data["title"].strip()

            # 1. Source Training Averages
            src_accs = {}
            for src, reps in data["results"].items():
                acc_only_reps = [
                    {ds: m["Accuracy"] for ds, m in r.items()} for r in reps
                ]
                src_accs[src] = pd.DataFrame(acc_only_reps).mean()
            df_src = pd.DataFrame(src_accs).T

            # 2. Fine-Tuning Averages
            ft_records = []
            for src, targets in data["fine_tuning_results"].items():
                for tgt, reps in targets.items():
                    acc_only_reps = [
                        {ds: m["Accuracy"] for ds, m in r.items()} for r in reps
                    ]
                    row = pd.DataFrame(acc_only_reps).mean().to_dict()
                    row["Source"] = src
                    row["FT_Target"] = tgt
                    ft_records.append(row)

            if not ft_records:
                continue
            df_ft = pd.DataFrame(ft_records).set_index(["Source", "FT_Target"])

            # 3. Construct Tables
            rows_absolute = []
            rows_diff = []
            for src in datasets:
                for tgt in [d for d in datasets if d != src]:
                    # Values
                    t_mfpt, t_cwru, t_kaist = (
                        df_src.loc[src, "mfpt"],
                        df_src.loc[src, "cwru"],
                        df_src.loc[src, "kaist"],
                    )
                    f_mfpt, f_cwru, f_kaist = (
                        df_ft.loc[(src, tgt), "mfpt"],
                        df_ft.loc[(src, tgt), "cwru"],
                        df_ft.loc[(src, tgt), "kaist"],
                    )

                    base_info = {"Source": src.upper(), "FT_Target": tgt.upper()}

                    # Absolute Data
                    rows_absolute.append(
                        {
                            **base_info,
                            "train_mfpt": t_mfpt,
                            "train_cwru": t_cwru,
                            "train_kaist": t_kaist,
                            "ft_mfpt": f_mfpt,
                            "ft_cwru": f_cwru,
                            "ft_kaist": f_kaist,
                        }
                    )

                    # Difference Data (FT - Training)
                    rows_diff.append(
                        {
                            **base_info,
                            "diff_mfpt": f_mfpt - t_mfpt,
                            "diff_cwru": f_cwru - t_cwru,
                            "diff_kaist": f_kaist - t_kaist,
                        }
                    )

            routines_dict[title] = {
                "absolute": pd.DataFrame(rows_absolute),
                "difference": pd.DataFrame(rows_diff),
            }
    return routines_dict


def write_grid_to_sheet(worksheet, df_dict, layout, mode, start_row_offset, workbook):
    # Styles
    title_fmt = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#305496",
            "font_color": "white",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    header_fmt = workbook.add_format(
        {
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bg_color": "#D9E1F2",
        }
    )
    cell_fmt = workbook.add_format(
        {"border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"}
    )
    merge_fmt = workbook.add_format(
        {
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "bg_color": "#FFFFFF",
            "num_format": "0.0000",
        }
    )

    R_GRID_OFFSET, C_GRID_OFFSET = 16, 10

    for (grid_r, grid_c), title in layout.items():
        if title not in df_dict:
            continue
        df = df_dict[title][mode]
        s_row, s_col = (
            (grid_r * R_GRID_OFFSET) + start_row_offset,
            grid_c * C_GRID_OFFSET,
        )

        # Headers
        worksheet.merge_range(
            s_row,
            s_col,
            s_row,
            s_col + (7 if mode == "absolute" else 4),
            title,
            title_fmt,
        )

        if mode == "absolute":
            worksheet.merge_range(
                s_row + 1, s_col, s_row + 1, s_col, "Info", header_fmt
            )
            worksheet.merge_range(
                s_row + 1, s_col + 1, s_row + 1, s_col + 3, "Training Phase", header_fmt
            )
            worksheet.merge_range(
                s_row + 1, s_col + 4, s_row + 1, s_col + 4, "Info", header_fmt
            )
            worksheet.merge_range(
                s_row + 1,
                s_col + 5,
                s_row + 1,
                s_col + 7,
                "Fine-Tuning Phase",
                header_fmt,
            )
            cols = [
                "Source",
                "MFPT",
                "CWRU",
                "KAIST",
                "FT Target",
                "MFPT",
                "CWRU",
                "KAIST",
            ]
        else:
            worksheet.merge_range(
                s_row + 1, s_col, s_row + 1, s_col + 1, "Info", header_fmt
            )
            worksheet.merge_range(
                s_row + 1,
                s_col + 2,
                s_row + 1,
                s_col + 4,
                "Fine-Tuning Improvement (Δ)",
                header_fmt,
            )
            cols = ["Source", "FT Target", "MFPT", "CWRU", "KAIST"]

        for i, col_name in enumerate(cols):
            worksheet.write(s_row + 2, s_col + i, col_name, header_fmt)

        # Data
        data_start = s_row + 3
        for i in range(len(df)):
            for j in range(len(cols)):
                val = df.iloc[i, j]
                worksheet.write(data_start + i, s_col + j, val, cell_fmt)

        # Merging Source (and Training if absolute)
        for i in range(0, len(df), 2):
            if i + 1 < len(df):
                worksheet.merge_range(
                    data_start + i,
                    s_col,
                    data_start + i + 1,
                    s_col,
                    df.iloc[i, 0],
                    merge_fmt,
                )
                if mode == "absolute":
                    for c in range(1, 4):
                        worksheet.merge_range(
                            data_start + i,
                            s_col + c,
                            data_start + i + 1,
                            s_col + c,
                            df.iloc[i, c],
                            merge_fmt,
                        )

        # Heatmap
        h_ranges = [(1, 3), (5, 7)] if mode == "absolute" else [(2, 4)]
        for r_start, r_end in h_ranges:
            worksheet.conditional_format(
                data_start,
                s_col + r_start,
                data_start + len(df) - 1,
                s_col + r_end,
                {
                    "type": "3_color_scale",
                    "min_color": "#F8696B",
                    "mid_color": "#FFEB84",
                    "max_color": "#63BE7B",
                    "min_type": "num",
                    "min_value": -0.5 if mode == "difference" else 0,
                    "mid_type": "num",
                    "mid_value": 0 if mode == "difference" else 0.5,
                    "max_type": "num",
                    "max_value": 0.5 if mode == "difference" else 1,
                },
            )


def export_to_xlsx(routines_dict, layout, output_path, source_filename):
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    workbook = writer.book
    src_fmt = workbook.add_format(
        {"bold": True, "font_size": 14, "font_color": "#44546A"}
    )

    for sheet_name, mode in [("Results", "absolute"), ("Difference", "difference")]:
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.write(0, 0, f"Source File: {source_filename}", src_fmt)
        write_grid_to_sheet(worksheet, routines_dict, layout, mode, 2, workbook)
        worksheet.set_column(0, 50, 12)

    writer.close()


if __name__ == "__main__":
    files = [
        "experiments/notebooks/results/full_results.jsonl",
        "experiments/notebooks/results_p_01/full_results_p_01.jsonl",
        "experiments/notebooks/results_p_005/full_results_p_005.jsonl",
        "experiments/notebooks/results_p_001/full_results_p_001.jsonl",
    ]
    output_dir = "experiments/notebooks/summarized_results/"
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
            os.path.join(output_dir, f"Summary{p_suffix}.xlsx"),
            path.name,
        )

    print("\nProcessing complete.")
