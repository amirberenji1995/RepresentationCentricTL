import os
import json
import pandas as pd
import numpy as np
from core.experiment_result import ExperimentResult
import xlsxwriter


def load_results_safely(directory_path="results"):
    all_results = {}
    if not os.path.exists(directory_path):
        return all_results

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(directory_path, file_name)
            try:
                with open(file_path, "r") as f:
                    line = f.readline()
                    if not line:
                        continue
                    data = json.loads(line)

                    # --- CRITICAL FIX: Clean the data before creating the object ---
                    # 1. Remove timing from results to avoid breaking acc_dfs property
                    if "results" in data and "timing_metadata" in data["results"]:
                        # We extract it to use later, then remove from the dict
                        timing = data["results"].pop("timing_metadata")
                        data["timing_data_extracted"] = timing

                    # 2. Reconstruct using model_construct to bypass strict List validation
                    exp_result = ExperimentResult.model_construct(**data)

                    routine_key = file_name.replace(".jsonl", "")
                    all_results[routine_key] = exp_result
            except Exception as e:
                print(f"Skipping {file_name} due to error: {e}")
    return all_results


def generate_experiment_summary_excel(
    all_results, output_filename="Experiment_Summary.xlsx"
):
    """
    Generates a formatted Excel spreadsheet with source and fine-tuning results,
    including a 3-color scale (Red-Orange-Green).
    """

    # Create an Excel writer with xlsxwriter engine
    writer = pd.ExcelWriter(output_filename, engine="xlsxwriter")
    workbook = writer.book
    worksheet = workbook.add_worksheet("Summary")

    # --- Formats ---
    header_format = workbook.add_format(
        {
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "bg_color": "#D9D9D9",
        }
    )
    title_format = workbook.add_format(
        {"bold": True, "size": 14, "align": "left", "valign": "vcenter"}
    )
    label_format = workbook.add_format({"bold": True, "border": 1})
    data_format = workbook.add_format({"num_format": "0.0000", "border": 1})

    datasets = ["mfpt", "cwru", "kaist"]
    col_offset = 0  # To move horizontally for each routine

    for routine_key, exp_result in all_results.items():
        # Start at row 0 for this routine's block
        row = 0

        # 1. Routine Title
        worksheet.write(row, col_offset, exp_result.title, title_format)
        row += 1

        # 2. Section Headers (Training vs Finetuning)
        worksheet.merge_range(
            row, col_offset, row, col_offset + 3, "Training", header_format
        )
        worksheet.merge_range(
            row, col_offset + 5, row, col_offset + 8, "Finetuning", header_format
        )
        row += 1

        # 3. Sub-headers
        worksheet.write(row, col_offset, "Training Set", header_format)
        worksheet.write(row, col_offset + 1, "Testing Set", header_format)
        worksheet.write(row, col_offset + 5, "Finetuning Set", header_format)
        worksheet.write(row, col_offset + 6, "Testing Set", header_format)
        row += 1

        # 4. Metric Columns (MFPT, CWRU, KAIST)
        for i, ds in enumerate(datasets):
            worksheet.write(row, col_offset + 1 + i, ds.upper(), header_format)
            worksheet.write(row, col_offset + 6 + i, ds.upper(), header_format)
        row += 1

        # --- DATA EXTRACTION ---
        # 5. Source Training Data (Mean)
        source_start_row = row
        mean_src_df = exp_result.mean_accs  # Assuming DataFrame: rows=train, cols=test
        for i, src_ds in enumerate(datasets):
            worksheet.write(row + i * 2, col_offset, src_ds.upper(), label_format)
            for j, test_ds in enumerate(datasets):
                val = mean_src_df.loc[src_ds, test_ds]
                worksheet.write(row + i * 2, col_offset + 1 + j, val, data_format)

        # 6. Fine-Tuning Data (Mean)
        ft_data = exp_result.fine_tuning_mean_accs
        ft_row = row
        for src_ds in datasets:
            ft_dict = ft_data.get(src_ds, {})
            # We skip the case where target == source, so we get 2 rows per source
            for tgt_ds in [d for d in datasets if d != src_ds]:
                worksheet.write(ft_row, col_offset + 5, tgt_ds.upper(), label_format)
                # The series contains accuracies for all test sets
                series = ft_dict.get(tgt_ds)
                for j, test_ds in enumerate(datasets):
                    val = series[test_ds]
                    worksheet.write(ft_row, col_offset + 6 + j, val, data_format)
                ft_row += 1

        # --- CONDITIONAL FORMATTING (Red -> Orange -> Green) ---
        # Define ranges for source (3x3 area) and FT (6x3 area)
        src_range = f"{xlsxwriter.utility.xl_rowcol_to_cell(row, col_offset + 1)}:{xlsxwriter.utility.xl_rowcol_to_cell(row + 5, col_offset + 3)}"
        ft_range = f"{xlsxwriter.utility.xl_rowcol_to_cell(row, col_offset + 6)}:{xlsxwriter.utility.xl_rowcol_to_cell(row + 5, col_offset + 8)}"

        color_scale_rule = {
            "type": "3_color_scale",
            "min_type": "num",
            "min_value": 0,
            "min_color": "#FF0000",  # Red
            "mid_type": "num",
            "mid_value": 0.5,
            "mid_color": "#FFA500",  # Orange
            "max_type": "num",
            "max_value": 1,
            "max_color": "#00FF00",  # Green
        }

        worksheet.conditional_format(src_range, color_scale_rule)
        worksheet.conditional_format(ft_range, color_scale_rule)

        # Move to the next routine (gap of 10 columns)
        col_offset += 10

    writer.close()
    print(f"Successfully saved results to {output_filename}")

    return output_filename
