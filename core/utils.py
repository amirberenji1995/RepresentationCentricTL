import os
import subprocess
import optuna
import pandas as pd
from typing import Callable, List, Literal, Dict, Any, Optional, Tuple, Type
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from pydantic import BaseModel
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
import random
from easy_torchkit.src.classification import ClassificationModel
from scipy.signal import resample
from damavand.damavand.utils import splitter
import json


colors = ["red", "yellow", "green"]
ryg_cmap = LinearSegmentedColormap.from_list("custom_ryg", colors)


def plot_df_as_heatmap(
    df: pd.DataFrame,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str | Colormap = ryg_cmap,
    fmt: str = ".4f",
    figsize: Tuple[int, int] = (6, 5),
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> None:
    plt.figure(figsize=figsize)

    sns.heatmap(df, vmin=vmin, vmax=vmax, cmap=cmap, fmt=fmt, annot=True, cbar=False)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def setup_github_repo(
    github_url: str,
    repo_name: str,
    base_dir: str,
    req_file: str,
) -> None:
    """
    Clone or update a GitHub repository and install its dependencies if available.

    Args:
        github_url (str): HTTPS or SSH URL of the GitHub repository.
        repo_name (str): Folder name the repository will be cloned into.
        base_dir (str): Parent directory where the repository will be placed.

    Returns:
        None
    """

    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)

    repo_path = os.path.join(base_dir, repo_name)

    # Clone or pull repository
    if os.path.isdir(repo_path):
        print(f"📁 Repository already exists at {repo_path}")
        print("🔄 Pulling latest changes...")
        subprocess.run(["git", "-C", repo_path, "pull"], check=True)
    else:
        print(f"⬇️ Cloning repository into {repo_path}...")
        subprocess.run(["git", "clone", github_url, repo_path], check=True)
        print("✔️ Clone complete.")

    # Install dependencies if requirements.txt exists
    # req_file = os.path.join(base_dir, "requirements.txt")

    if os.path.isfile(req_file):
        print("\n📦 Installing dependencies from requirements.txt ...")
        subprocess.run(["pip", "install", "-r", req_file], check=True)
        print("✔️ Dependencies installed.")
    else:
        print("\n⚠️ No requirements.txt found, skipping dependency installation.")


def device_recognizer() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU Runtime Detected")

    else:
        device = torch.device("cpu")
        print("No GPU Found - CPU Runtime")

    return device


def datasets_loader(
    names: List[str] | Literal["all"] = "all",
) -> Dict[str, pd.DataFrame]:
    datasets = {
        "mfpt": {
            "path": "/nfs/home/amiber/phme26/mfpt_data.csv",
            "index_col": 0,
        },
        "cwru": {
            "path": "/nfs/home/amiber/phme26/cwru_data.csv",
            "index_col": 0,
        },
        "kaist": {
            "path": "/nfs/home/amiber/phme26/kaist_data.csv",
            "index_col": 0,
        },
    }

    if names == "all":
        names = datasets.keys()

    return {
        key: pd.read_csv(
            datasets[key]["path"],
            index_col=datasets[key]["index_col"],
            low_memory=False,
        )
        for key in names
    }


def label_unifier(
    mfpt_data: pd.DataFrame, cwru_data: pd.DataFrame, kaist_data: pd.DataFrame
) -> List[pd.DataFrame]:
    cwru_data["state"] = cwru_data["state"].replace(["OR@6", "OR@12", "OR@3"], "OR")
    cwru_data["state"] = cwru_data["state"].replace("normal", "Normal")

    kaist_data["state"] = kaist_data["state"].replace("BPFI", "IR")
    kaist_data["state"] = kaist_data["state"].replace("BPFO", "OR")

    return mfpt_data, cwru_data, kaist_data


def signal_target_decleration(
    mfpt_data: pd.DataFrame, cwru_data: pd.DataFrame, kaist_data: pd.DataFrame
) -> List[pd.DataFrame]:
    return (
        mfpt_data.iloc[:, :-4],
        mfpt_data["state"],
        cwru_data.iloc[:, :-5],
        cwru_data["state"],
        kaist_data.iloc[:, :-3],
        kaist_data["state"],
    )


def train_test_splitter(
    signal_target_dict: Dict[str, Dict[str, pd.DataFrame]],
    test_size: float = 0.4,
    random_state: int = 42,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    output_dict = {}
    for key in signal_target_dict.keys():
        temp_dict = {}
        (
            temp_dict["train_x"],
            temp_dict["test_x"],
            temp_dict["train_y"],
            temp_dict["test_y"],
        ) = train_test_split(
            signal_target_dict[key]["x"],
            signal_target_dict[key]["y"],
            test_size=test_size,
            random_state=random_state,
        )

        output_dict[key] = temp_dict

    return output_dict


def label_encoder(labels: List[pd.Series]) -> List[pd.Series]:
    le = LabelEncoder()
    le.fit(labels[0])

    return [le.transform(label) for label in labels]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fine_tuning_summary_matrices(fine_tuning_mean_accs):
    matrices = {}
    for training_set, fine_tune_dict in fine_tuning_mean_accs.items():
        df = pd.DataFrame(fine_tune_dict).T

        df.index.name = "Fine-Tuning Set"
        df.columns.name = "Test Set"

        matrices[training_set] = df

    return matrices


def p_subsampler_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    rnd_state: int,
    p: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.size(0)
    k = int(n * p)

    gen = torch.Generator(device=x.device)
    gen.manual_seed(rnd_state)

    idx = torch.randperm(n, generator=gen, device=x.device)[:k]

    return x[idx], y[idx]


class Routine(BaseModel):
    description: str
    data_processing: List[Tuple[Callable, Dict[str, Any]]]
    model: Tuple[Type[ClassificationModel], Dict[str, Any]]
    training_best_params_ranges: Optional[Dict]
    fine_tuning_best_params_ranges: Optional[Dict] | None = None

    class Config:
        arbitrary_types_allowed = True

    def apply_preprocessing(self, data, dataset_name, src_dataset_name=None):
        processed_data = data
        for func, all_kwargs in self.data_processing:
            # Determine which parameter set to use
            # CRITICAL: Sequence length must follow the Source for Transfer Learning
            use_source_logic = src_dataset_name and func.__name__ in [
                "resampler",
                "sequence_data",
            ]

            lookup_key = src_dataset_name if use_source_logic else dataset_name

            if func.__name__ == "sequence_data":
                current_kwargs = all_kwargs.get(self.description, {}).get(
                    lookup_key, {}
                )
            else:
                current_kwargs = all_kwargs.get(lookup_key, {})

            processed_data = func(processed_data, **current_kwargs)

        return processed_data


class BestParams(BaseModel):
    description: str
    training_best_params: Dict
    fine_tuning_best_params: Optional[Dict] = None
    random_state: int
    training_study_details: Optional[List[Dict[str, Any]]] = None
    fine_tuning_study_details: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed = True

    def log_to_jsonl(
        self, log_file="best_params.jsonl", exclude: Optional[List[str]] = None
    ):
        exclude_set = set(exclude) if exclude else None

        json_line = self.model_dump_json(exclude=exclude_set)

        with open(log_file, "a") as f:
            f.write(json_line + "\n")


def extract_study_data(study: optuna.Study) -> List[Dict[str, Any]]:
    """Converts optuna trials into a JSON-serializable list of dicts."""
    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "state": str(trial.state),
                "datetime_start": trial.datetime_start.isoformat()
                if trial.datetime_start
                else None,
                "datetime_complete": trial.datetime_complete.isoformat()
                if trial.datetime_complete
                else None,
            }
        )
    return trials_data


def resampler(
    signals: np.array,
    desired_length: int,
    method: Literal["linear", "fourier"] = "fourier",
):
    if method == "linear":
        raise NotImplementedError
    elif method == "fourier":
        return resample(signals, desired_length, axis=1)
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def sequence_data(data, win_len=400, hop_len=400):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # Generate the sequenced data
    res = np.array([splitter(row, win_len=win_len, hop_len=hop_len) for row in data])

    # FIX: If res is (Samples, Win_Len), expand it to (Samples, 1, Win_Len)
    if res.ndim == 2:
        res = res[:, np.newaxis, :]

    return res


def load_best_params_lookup(file_path="best_params.jsonl"):
    """Creates a lookup table: {routine_description: param_data}"""
    lookup = {}
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            lookup[data["description"]] = data
    return lookup


def torch_sampler(
    x: torch.Tensor,
    y: torch.Tensor,
    rnd_state: int,
    subsampling_style: Literal["percentage", "shots_per_class"],
    subsampling_factor: float,
):
    def p_subsampler_torch(
        x: torch.Tensor,
        y: torch.Tensor,
        rnd_state: int,
        p: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        k = int(n * p)

        gen = torch.Generator(device=x.device)
        gen.manual_seed(rnd_state)

        idx = torch.randperm(n, generator=gen, device=x.device)[:k]

        return x[idx], y[idx]

    def s_subsampler_torch(
        x: torch.Tensor,
        y: torch.Tensor,
        rnd_state: int,
        shots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_classes = torch.unique(y)
        gen = torch.Generator(device=x.device).manual_seed(rnd_state)

        indices = []
        for c in unique_classes:
            # Get indices for the current class
            cls_indices = (y == c).nonzero(as_tuple=True)[0]

            # Shuffle and pick the first 'n' shots
            perm = torch.randperm(len(cls_indices), generator=gen, device=x.device)
            indices.append(cls_indices[perm[:shots]])

        # Concatenate and sort to preserve original order (optional but cleaner)
        final_idx = torch.cat(indices).sort()[0]
        return x[final_idx], y[final_idx]

    if subsampling_style == "percentage":
        return p_subsampler_torch(x, y, rnd_state, subsampling_factor)
    elif subsampling_style == "shots_per_class":
        return s_subsampler_torch(x, y, rnd_state, int(subsampling_factor))
    else:
        raise ValueError(f"Invalid style: {subsampling_style}")
