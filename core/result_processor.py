import pandas as pd
import json


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

            # 1. Source Training Averages (Accuracy)
            src_accs = {
                src: pd.DataFrame(
                    [{ds: m["accuracy"] for ds, m in r.items()} for r in reps]
                ).mean()
                for src, reps in data["results"].items()
            }
            df_src = pd.DataFrame(src_accs).T

            # 2. Fine-Tuning Averages (Accuracy)
            ft_records = []
            for src, targets in data["fine_tuning_results"].items():
                for tgt, reps in targets.items():
                    row = (
                        pd.DataFrame(
                            [{ds: m["accuracy"] for ds, m in r.items()} for r in reps]
                        )
                        .mean()
                        .to_dict()
                    )
                    row.update({"Source": src, "FT_Target": tgt})
                    ft_records.append(row)
            df_ft = pd.DataFrame(ft_records).set_index(["Source", "FT_Target"])

            # 3. Timing Averages
            src_time = {
                src: pd.DataFrame(reps).mean()
                for src, reps in data.get("source_timing_raw", {}).items()
            }
            df_src_time = pd.DataFrame(src_time).T

            ft_time_recs = []
            for src, targets in data.get("fine_tuning_timing_raw", {}).items():
                for tgt, reps in targets.items():
                    row = pd.DataFrame(reps).mean().to_dict()
                    row.update({"Source": src, "FT_Target": tgt})
                    ft_time_recs.append(row)
            df_ft_time = pd.DataFrame(ft_time_recs).set_index(["Source", "FT_Target"])

            # 4. Construct Tables
            rows_perf = []
            rows_time = []
            for src in datasets:
                for tgt in [d for d in datasets if d != src]:
                    # Performance Row (Fixed Column Alignment)
                    rows_perf.append(
                        {
                            "Source": src.upper(),
                            "T_MFPT": df_src.loc[src, "mfpt"],
                            "T_CWRU": df_src.loc[src, "cwru"],
                            "T_KAIST": df_src.loc[src, "kaist"],
                            "FT_Target": tgt.upper(),
                            "Abs_MFPT": df_ft.loc[(src, tgt), "mfpt"],
                            "Abs_CWRU": df_ft.loc[(src, tgt), "cwru"],
                            "Abs_KAIST": df_ft.loc[(src, tgt), "kaist"],
                            "Diff_MFPT": df_ft.loc[(src, tgt), "mfpt"]
                            - df_src.loc[src, "mfpt"],
                            "Diff_CWRU": df_ft.loc[(src, tgt), "cwru"]
                            - df_src.loc[src, "cwru"],
                            "Diff_KAIST": df_ft.loc[(src, tgt), "kaist"]
                            - df_src.loc[src, "kaist"],
                        }
                    )
                    # Timing Row
                    rows_time.append(
                        {
                            "Source": src.upper(),
                            "Src_Total": df_src_time.loc[src, "total_time"]
                            if src in df_src_time.index
                            else 0,
                            "Src_Avg": df_src_time.loc[src, "average_epoch_time"]
                            if src in df_src_time.index
                            else 0,
                            "FT_Target": tgt.upper(),
                            "FT_Total": df_ft_time.loc[(src, tgt), "total_time"]
                            if (src, tgt) in df_ft_time.index
                            else 0,
                            "FT_Avg": df_ft_time.loc[(src, tgt), "average_epoch_time"]
                            if (src, tgt) in df_ft_time.index
                            else 0,
                        }
                    )

            routines_dict[title] = {
                "performance": pd.DataFrame(rows_perf),
                "timing": pd.DataFrame(rows_time),
            }
    return routines_dict


def write_grid_to_sheet(worksheet, df_dict, layout, mode, start_row_offset, workbook):
    # Colors
    c_red, c_orange, c_green = "#F8696B", "#FBAF5D", "#63BE7B"

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

    R_GRID_OFFSET = 18 if mode == "performance" else 15
    C_GRID_OFFSET = 12 if mode == "performance" else 8

    for (grid_r, grid_c), title in layout.items():
        if title not in df_dict:
            continue
        df = df_dict[title][mode]
        s_row, s_col = (
            (grid_r * R_GRID_OFFSET) + start_row_offset,
            grid_c * C_GRID_OFFSET,
        )

        if mode == "performance":
            worksheet.merge_range(s_row, s_col, s_row, s_col + 10, title, title_fmt)
            # Level 1
            worksheet.write(s_row + 1, s_col, "Info", header_fmt)
            worksheet.merge_range(
                s_row + 1, s_col + 1, s_row + 1, s_col + 3, "Training Phase", header_fmt
            )
            worksheet.write(s_row + 1, s_col + 4, "Info", header_fmt)
            worksheet.merge_range(
                s_row + 1,
                s_col + 5,
                s_row + 1,
                s_col + 10,
                "Fine-Tuning Phase",
                header_fmt,
            )
            # Level 2
            worksheet.merge_range(
                s_row + 2,
                s_col + 5,
                s_row + 2,
                s_col + 7,
                "Absolute Accuracy",
                header_fmt,
            )
            worksheet.merge_range(
                s_row + 2,
                s_col + 8,
                s_row + 2,
                s_col + 10,
                "Improvement (Δ)",
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
                "MFPT",
                "CWRU",
                "KAIST",
            ]
            header_row = s_row + 3
        else:
            worksheet.merge_range(s_row, s_col, s_row, s_col + 5, title, title_fmt)
            worksheet.write(s_row + 1, s_col, "Info", header_fmt)
            worksheet.merge_range(
                s_row + 1,
                s_col + 1,
                s_row + 1,
                s_col + 2,
                "Source Training Time",
                header_fmt,
            )
            worksheet.write(s_row + 1, s_col + 3, "Info", header_fmt)
            worksheet.merge_range(
                s_row + 1,
                s_col + 4,
                s_row + 1,
                s_col + 5,
                "Fine-Tuning Time",
                header_fmt,
            )
            cols = ["Source", "Total (s)", "Avg/Ep", "FT Target", "Total (s)", "Avg/Ep"]
            header_row = s_row + 2

        for i, col_name in enumerate(cols):
            worksheet.write(header_row, s_col + i, col_name, header_fmt)

        data_start = header_row + 1
        for i in range(len(df)):
            for j in range(len(cols)):
                worksheet.write(data_start + i, s_col + j, df.iloc[i, j], cell_fmt)

        # Merge Blocks
        for i in range(0, len(df), 2):
            worksheet.merge_range(
                data_start + i,
                s_col,
                data_start + i + 1,
                s_col,
                df.iloc[i, 0],
                merge_fmt,
            )
            if mode == "performance":
                for c in range(1, 4):
                    worksheet.merge_range(
                        data_start + i,
                        s_col + c,
                        data_start + i + 1,
                        s_col + c,
                        df.iloc[i, c],
                        merge_fmt,
                    )
            else:
                for c in range(1, 3):
                    worksheet.merge_range(
                        data_start + i,
                        s_col + c,
                        data_start + i + 1,
                        s_col + c,
                        df.iloc[i, c],
                        merge_fmt,
                    )

        # Aesthetic Heatmaps
        if mode == "performance":
            # 1) Absolute Ranges (0, 0.5, 1)
            for r_start, r_end in [(1, 3), (5, 7)]:
                worksheet.conditional_format(
                    data_start,
                    s_col + r_start,
                    data_start + 5,
                    s_col + r_end,
                    {
                        "type": "3_color_scale",
                        "min_color": c_red,
                        "mid_color": c_orange,
                        "max_color": c_green,
                        "min_type": "num",
                        "min_value": 0,
                        "mid_type": "num",
                        "mid_value": 0.5,
                        "max_type": "num",
                        "max_value": 1,
                    },
                )
            # 2) Delta Ranges (-0.5, 0, 0.5)
            worksheet.conditional_format(
                data_start,
                s_col + 8,
                data_start + 5,
                s_col + 10,
                {
                    "type": "3_color_scale",
                    "min_color": c_red,
                    "mid_color": c_orange,
                    "max_color": c_green,
                    "min_type": "num",
                    "min_value": -1,
                    "mid_type": "num",
                    "mid_value": 0,
                    "max_type": "num",
                    "max_value": 1,
                },
            )


def export_to_xlsx(routines_dict, layout, output_path, source_filename):
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    workbook = writer.book
    src_fmt = workbook.add_format(
        {"bold": True, "font_size": 12, "font_color": "#44546A"}
    )

    for sheet_name, mode in [("Performance", "performance"), ("Timing", "timing")]:
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.write(0, 0, f"Source File: {source_filename}", src_fmt)
        write_grid_to_sheet(worksheet, routines_dict, layout, mode, 2, workbook)
        worksheet.set_column(0, 100, 12)
    writer.close()
