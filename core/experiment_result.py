from enum import StrEnum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Literal, Optional
import numpy as np
import pandas as pd
from core.utils import plot_df_as_heatmap


class ExperimentType(StrEnum):
    TRAINING = "training"
    FINE_TUNING = "fine_tuning"


class ExperimentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reps: int = 5
    title: str | None = None
    random_states: List[int] = Field(default_factory=list)
    results: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)
    experiment_type: ExperimentType = Field(default=ExperimentType.TRAINING)
    fine_tuning_results: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    fine_tuning_models: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        if not self.random_states:
            self.random_states = np.random.randint(0, 100, self.reps).tolist()

    def log_to_jsonl(
        self, log_file="best_params.jsonl", exclude: Optional[List[str]] = None
    ):
        exclude_set = set(exclude) if exclude else None

        json_line = self.model_dump_json(exclude=exclude_set)

        with open(log_file, "a") as f:
            f.write(json_line + "\n")

    @property
    def acc_dfs(self) -> Dict[str, pd.DataFrame]:
        """Summarizes raw results. Each DF has Test Sets as columns, Reps as rows."""
        if not self.results:
            return {}
        return {
            train_set: pd.DataFrame(
                [
                    {test_set: m["Accuracy"] for test_set, m in run.items()}
                    for run in runs
                ]
            )
            for train_set, runs in self.results.items()
        }

    @property
    def mean_accs(self) -> pd.DataFrame:
        return pd.DataFrame({k: df.mean() for k, df in self.acc_dfs.items()}).T

    @property
    def std_accs(self) -> pd.DataFrame:
        return pd.DataFrame({k: df.std() for k, df in self.acc_dfs.items()}).T

    def plot_heatmap(
        self,
        metric: Literal["mean", "std"] = "mean",
        **kwargs,
    ) -> None:
        df = self.mean_accs if metric == "mean" else self.std_accs

        kwargs.setdefault("x_label", "Testing Set")
        kwargs.setdefault("y_label", "Training Set")

        if "title" not in kwargs:
            base_title = f"Accuracy ({metric.capitalize()} over {self.reps} runs)"
            kwargs["title"] = (
                f"{self.title} - {base_title}" if self.title else base_title
            )

        plot_df_as_heatmap(df, **kwargs)

    @property
    def fine_tuning_acc_dfs(self):
        fine_tuning_dfs = {}

        results_dict = self.fine_tuning_results
        training_sets = results_dict.keys()

        for training_set in training_sets:
            fine_tuning_dfs[training_set] = {}
            fine_tuning_targets = [
                item for item in training_sets if item != training_set
            ]

            for fine_tuning_set in fine_tuning_targets:
                data_for_df = []

                reps_list = results_dict[training_set][fine_tuning_set]

                for rep_results in reps_list:
                    acc_only = {
                        test_set: metrics["Accuracy"]
                        for test_set, metrics in rep_results.items()
                    }
                    data_for_df.append(acc_only)

                df = pd.DataFrame(data_for_df)
                fine_tuning_dfs[training_set][fine_tuning_set] = df

        return fine_tuning_dfs

    @property
    def fine_tuning_mean_accs(self):
        mean_fine_tuning_dfs = {}

        for training_set in self.fine_tuning_acc_dfs.keys():
            mean_fine_tuning_dfs[training_set] = {}
            for fine_tuning_set in self.fine_tuning_acc_dfs[training_set].keys():
                mean_fine_tuning_dfs[training_set][fine_tuning_set] = (
                    self.fine_tuning_acc_dfs[training_set][fine_tuning_set].mean()
                )

        return mean_fine_tuning_dfs

    @property
    def fine_tuning_std_accs(self):
        std_fine_tuning_dfs = {}

        for training_set in self.fine_tuning_acc_dfs.keys():
            std_fine_tuning_dfs[training_set] = {}
            for fine_tuning_set in self.fine_tuning_acc_dfs[training_set].keys():
                std_fine_tuning_dfs[training_set][fine_tuning_set] = (
                    self.fine_tuning_acc_dfs[training_set][fine_tuning_set].std()
                )

        return std_fine_tuning_dfs

    def get_training_time_summary(
        self, format: Literal["dictionary", "dataframe"] = "dictionary"
    ):
        timing_summary = {}

        for ds in self.models.keys():
            timing_summary[ds] = []
            for item in self.models[ds]:
                timing_summary[ds].append(
                    {
                        "total_time": item.history[-1].total_time,
                        "average_epoch_time": item.history[-1].average_epoch_time,
                    }
                )

        if format == "dictionary":
            return timing_summary
        elif format == "dataframe":
            return pd.DataFrame(
                [
                    {
                        "training_dataset": dataset_name.upper(),
                        "total_time": entry["total_time"],
                        "average_epoch_time": entry["average_epoch_time"],
                    }
                    for dataset_name, entries in timing_summary.items()
                    for entry in entries
                ]
            )
        else:
            raise ValueError(f"Invalid format: {format}")

    @property
    def training_time_mean(self):
        # Now calling it as a method with ()
        return (
            self.get_training_time_summary(format="dataframe")
            .groupby("training_dataset", as_index=False)
            .mean()
        )

    def get_fine_tuning_time_summary(
        self, format: Literal["dictionary", "dataframe"] = "dictionary"
    ):
        timing_summary = {}
        for train_set in self.fine_tuning_models.keys():
            timing_summary[train_set] = {}
            for fine_tuning_set in self.fine_tuning_models[train_set].keys():
                timing_summary[train_set][fine_tuning_set] = []
                for item in self.fine_tuning_models[train_set][fine_tuning_set]:
                    timing_summary[train_set][fine_tuning_set].append(
                        {
                            "total_time": item.history[-1].total_time,
                            "average_epoch_time": item.history[-1].average_epoch_time,
                        }
                    )

        if format == "dictionary":
            return timing_summary
        elif format == "dataframe":
            return pd.DataFrame(
                [
                    {
                        "training_dataset": train_set.upper(),
                        "fine_tuning_dataset": fine_tuning_set.upper(),
                        "total_time": entry["total_time"],
                        "average_epoch_time": entry["average_epoch_time"],
                    }
                    for train_set, fine_tuning_sets in timing_summary.items()
                    for fine_tuning_set, entries in fine_tuning_sets.items()
                    for entry in entries
                ]
            )
        else:
            raise ValueError(f"Invalid format: {format}")

    @property
    def fine_tuning_time_mean(self):
        # Now calling it as a method with ()
        return (
            self.get_fine_tuning_time_summary(format="dataframe")
            .groupby(["training_dataset", "fine_tuning_dataset"], as_index=False)
            .mean()
        )
