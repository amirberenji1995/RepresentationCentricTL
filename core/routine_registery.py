from .model_repository import CNNClassifier, DNNClassifier, LSTMClassifier
from damavand.damavand.signal_processing.transformations import env, fft, zoomed_fft
import scipy
from damavand.damavand.utils import z_score_scaler
from core.utils import resampler, sequence_data
from numpy import expand_dims

fine_tuning_search_space = {
    "supervised": None,
    "contrastive": {
        "pairs_per_sample": [1, 2],
    },
    "dynamic_bootstrapping": {
        "warmup_epochs": [5],
        "bmm_iters": [10],
    },
}

general_best_params_ranges = {
    "mfpt": {
        "lr": [0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128, 256],
    },
    "cwru": {
        "lr": [0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128, 256],
    },
    "kaist": {
        "lr": [0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64, 128, 256],
    },
}


lstm_best_params_ranges = {
    "mfpt": {
        "lr": [0.0005, 0.0001, 0.00005, 0.00001],
        "batch_size": [8, 16, 32, 64, 128],
    },
    "cwru": {
        "lr": [0.0005, 0.0001, 0.00005, 0.00001],
        "batch_size": [8, 16, 32, 64, 128],
    },
    "kaist": {
        "lr": [0.0005, 0.0001, 0.00005, 0.00001],
        "batch_size": [8, 16, 32, 64, 128],
    },
}

scaler_args = {
    "mfpt": {"axis": 1, "return_df": False},
    "cwru": {"axis": 1, "return_df": False},
    "kaist": {"axis": 1, "return_df": False},
}

env_args = {"mfpt": {}, "cwru": {}, "kaist": {}}

fft_args = {
    "mfpt": {
        "window": scipy.signal.windows.hann(1, 9600),
        "freq_filter": scipy.signal.butter(
            25, [5, 23500], "bandpass", fs=48828, output="sos"
        ),
    },
    "cwru": {
        "window": scipy.signal.windows.hann(1, 2400),
        "freq_filter": scipy.signal.butter(
            25, [5, 5500], "bandpass", fs=12000, output="sos"
        ),
    },
    "kaist": {
        "window": scipy.signal.windows.hann(1, 4800),
        "freq_filter": scipy.signal.butter(
            25, [5, 12500], "bandpass", fs=25600, output="sos"
        ),
    },
}

zoomed_fft_args = {
    "mfpt": {
        "f_min": 0,
        "f_max": 1000,
        "desired_len": 1000,
        "sampling_freq": 48828,
        "window": scipy.signal.windows.hann(9600),
        "freq_filter": scipy.signal.butter(
            25, [5, 950], "bandpass", fs=48828, output="sos"
        ),
    },
    "cwru": {
        "f_min": 0,
        "f_max": 1000,
        "desired_len": 1000,
        "sampling_freq": 12000,
        "window": scipy.signal.windows.hann(2400),
        "freq_filter": scipy.signal.butter(
            25, [5, 950], "bandpass", fs=12000, output="sos"
        ),
    },
    "kaist": {
        "f_min": 0,
        "f_max": 1000,
        "desired_len": 1000,
        "sampling_freq": 25600,
        "window": scipy.signal.windows.hann(4800),
        "freq_filter": scipy.signal.butter(
            25, [5, 950], "bandpass", fs=25600, output="sos"
        ),
    },
}

fft_resampling_args = {
    "mfpt": {"desired_length": 4800},
    "cwru": {"desired_length": 1200},
    "kaist": {"desired_length": 2400},
}

fft_dnn_classifier_args = {
    "mfpt": {"input_size": 4800},
    "cwru": {"input_size": 1200},
    "kaist": {"input_size": 2400},
}

sequencer_args = {
    "Raw -> Scaled -> Sequenced -> LSTM": {
        "mfpt": {"win_len": 40, "hop_len": 40},
        "cwru": {"win_len": 40, "hop_len": 40},
        "kaist": {"win_len": 80, "hop_len": 80},
    },
    "Raw -> Env -> Scaled -> Sequenced -> LSTM": {
        "mfpt": {"win_len": 1200, "hop_len": 1200},
        "cwru": {"win_len": 20, "hop_len": 20},
        "kaist": {"win_len": 100, "hop_len": 100},
    },
}

dim_expand_args = {
    "mfpt": {"axis": 1},
    "cwru": {"axis": 1},
    "kaist": {"axis": 1},
}

ROUTINE_REGISTERY = {
    "raw->scaled->cnn": {
        "description": "Raw -> Scaled -> CNN",
        "data_processing": [
            (z_score_scaler, scaler_args),
            (expand_dims, dim_expand_args),
        ],
        "model": (CNNClassifier, {}),
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->env->scaled->cnn": {
        "description": "Raw -> Env -> Scaled -> CNN",
        "data_processing": [
            (env, env_args),
            (z_score_scaler, scaler_args),
            (expand_dims, dim_expand_args),
        ],
        "model": [CNNClassifier, {}],
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->fft->scaled->cnn": {
        "description": "Raw -> FFT -> Scaled -> CNN",
        "data_processing": [
            (fft, fft_args),
            (z_score_scaler, scaler_args),
            (expand_dims, dim_expand_args),
        ],
        "model": [CNNClassifier, {}],
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->env->fft->scaled->cnn": {
        "description": "Raw -> Env -> FFT -> Scaled -> CNN",
        "data_processing": [
            (env, env_args),
            (zoomed_fft, zoomed_fft_args),
            (z_score_scaler, scaler_args),
            (expand_dims, dim_expand_args),
        ],
        "model": [CNNClassifier, {}],
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->zoomedfft->scaled->cnn": {
        "description": "Raw -> Zoomed FFT -> Scaled -> CNN",
        "data_processing": [
            (fft, fft_args),
            (z_score_scaler, scaler_args),
            (expand_dims, dim_expand_args),
        ],
        "model": [CNNClassifier, {}],
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->env->zoomedfft->scaled->cnn": {
        "description": "Raw -> Env -> Zoomed FFT -> Scaled -> CNN",
        "data_processing": [
            (env, env_args),
            (zoomed_fft, zoomed_fft_args),
            (z_score_scaler, scaler_args),
            (expand_dims, dim_expand_args),
        ],
        "model": [CNNClassifier, {}],
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->zoomedfft->scaled->dnn": {
        "description": "Raw -> Zoomed FFT -> Scaled -> DNN",
        "data_processing": [
            (zoomed_fft, zoomed_fft_args),
            (z_score_scaler, scaler_args),
        ],
        "model": (DNNClassifier, {}),
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->env->zoomedfft->scaled->dnn": {
        "description": "Raw -> Env -> Zoomed FFT -> Scaled -> DNN",
        "data_processing": [
            (env, env_args),
            (zoomed_fft, zoomed_fft_args),
            (z_score_scaler, scaler_args),
        ],
        "model": (DNNClassifier, {}),
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->scaled->sequenced->lstm": {
        "description": "Raw -> Scaled -> Sequenced -> LSTM",
        "data_processing": [
            (z_score_scaler, scaler_args),
            (sequence_data, sequencer_args),
        ],
        "model": (
            LSTMClassifier,
            sequencer_args["Raw -> Scaled -> Sequenced -> LSTM"],
        ),
        "training_best_params_ranges": lstm_best_params_ranges,
        "fine_tuning_best_params_ranges": lstm_best_params_ranges,
    },
    "raw->env->scaled->sequenced->lstm": {
        "description": "Raw -> Env -> Scaled -> Sequenced -> LSTM",
        "data_processing": [
            (env, env_args),
            (z_score_scaler, scaler_args),
            (sequence_data, sequencer_args),
        ],
        "model": (
            LSTMClassifier,
            sequencer_args["Raw -> Env -> Scaled -> Sequenced -> LSTM"],
        ),
        "training_best_params_ranges": lstm_best_params_ranges,
        "fine_tuning_best_params_ranges": lstm_best_params_ranges,
    },
    "raw->fft->scaled->resampled->dnn": {
        "description": "Raw -> FFT -> Scaled -> Resampled -> DNN",
        "data_processing": [
            (fft, fft_args),
            (resampler, fft_resampling_args),
            (z_score_scaler, scaler_args),
        ],
        "model": (DNNClassifier, fft_dnn_classifier_args),
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
    "raw->env->fft->scaled->resampled->dnn": {
        "description": "Raw -> Env -> FFT -> Scaled -> Resampled -> DNN",
        "data_processing": [
            (env, env_args),
            (fft, fft_args),
            (resampler, fft_resampling_args),
            (z_score_scaler, scaler_args),
        ],
        "model": (DNNClassifier, fft_dnn_classifier_args),
        "training_best_params_ranges": general_best_params_ranges,
        "fine_tuning_best_params_ranges": general_best_params_ranges,
    },
}
