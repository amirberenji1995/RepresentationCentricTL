This is the official repository for:

**Evaluation of Input Presentation in Transfer Learning for Bearing Fault Detection**

submitted to the [PHME2026](https://phm-europe.org/) and currently under review.

## Research questions and findings

We conducted this study with the following research questions and finished it with the corresponding findings:

1. **Does preprocessing, affect transferability?**: Yes; both transferability and catastrophic forgetting show significant sensitivity towards the choice of preprocessing.

2. **Is data efficiency preprocessing-dependent?**: Dramatically; with as few as 30 samples, some preprocessing routines manage to achieve competitive results.

3. **Can we replace target labels with pseudo-labels?**: Not entirely; the target-set budget can be practically reduced, however, a complete replacement is not possible.

## Experiments

### Preprocessings included
In this study, we compare 12 preprocessing routines, as below:

| No. | Domain | Title | Demodulation | Frequency Transformation |
| :--- | :--- | :--- | :---: | :--- |
| 1 | **Time** | Raw → Scaling → CNN | ✘ | ✘ |
| 2 | **Time** | Raw → Env → Scaling → CNN | ✔ | ✘ |
| 3 | **Time** | Raw → Scaling → Sequencing → LSTM | ✘ | ✘ |
| 4 | **Time** | Raw → Env → Scaling → Sequencing → LSTM | ✔ | ✘ |
| 5 | **Frequency** | Raw → FFT → Scaling → CNN | ✘ | FFT |
| 6 | **Frequency** | Raw → Env → FFT → Scaling → CNN | ✔ | FFT |
| 7 | **Frequency** | Raw → FFT → Scaling → Resampling → DNN | ✘ | FFT |
| 8 | **Frequency** | Raw → Env → FFT → Scaling → Resampling → DNN | ✔ | FFT |
| 9 | **Frequency** | Raw → Zoomed FFT → Scaling → CNN | ✘ | Zoomed FFT |
| 10 | **Frequency** | Raw → Env → Zoomed FFT → Scaled → CNN | ✔ | Zoomed FFT |
| 11 | **Frequency** | Raw → Zoomed FFT → Scaling → DNN | ✘ | Zoomed FFT |
| 12 | **Frequency** | Raw → Env → Zoomed FFT → Scaling → DNN | ✔ | Zoomed FFT |

### Datasets
We conduct our experiments over 3 different benchmark datasets of bearing faults, as below:

|Dataset|Sampling Frequency (KHz)|
|-------|------------------------|
|MFPT|97.6|
|CWRU|12|
|KAIST|25.6|

### Experimental design
We follow a classical transfer learning paradigm across the three datasets above. Generally, our experiments involve train-
ing over one of the sets (source), followed by fine-tuning on the two remaining target datasets. Moreover, we investigate the following trasfer learning scenarios:

* **Direct transfer**: No fine-tuning at all; it only involves training a model on the source set and evaluating its performance over the source and the remaining sets, assessing pre-fine-tuning classification performance.
* **Labeled fine-tuning**: Treating the fine-tuning problem as a conventional supervised classification problem, we fine-tune the source model on a target set.
* **Fine-tuning on pseudo-labels**: To reduce the costs of transfer learning, by replacing ground-truth labels from the target set with pseudo-labels from the source model, we fine-tune the source model on the target-set with the pseduo-labels from the pre-fine-tuning source model. To encounter the label noise induced by using an imperfect oracle, we employ Dynamic Bootstrapping to fine-tune.

To fully address the research questions mentioned above, we implemented following sets of experiments:

* Comparing the direct classification performance of each of the routines on target datasets.
* Benchmarking the data efficiency of the routines by changing the amount of target-set data used for supervised fine-tuning.
* Investigating the feasibility of using the source-set model as an erroneous oracle to pseudo-label the unlabeled data from the target set and fine-tuning on it using DB.

## Results

In this section, I go over a small portion of our results, supporting our findings corresponding to the reasearch questions mentioned above:

1. Following figure, visualizes the pre-fine-tuning ”base” performance (yellow), the post-fine-tuning target accuracy (blue), and the relative performance degradation on the source set, commonly referred to as catastrophic forgetting (red). These results indicate that the choice of routine significantly influences the generalizability of learned features.

    <img src="experiments/notebooks/results/polar_charts/_fs_supervised_ss_None_sp_None.png" alt="Alt Text" height="700">

2. To evaluate the data-efficiency of the pipelines, we repeated the same experiments for different volumes of the fine-tuning data; following figure includes performance for the case 10 shots-per-class is visualized. Although the the fine-tuning set consists only 30 samples, still a number of the routines manage to get competitive results for some of the source-target pairs.

    <img src="experiments/notebooks/results/polar_charts/_fs_supervised_ss_shots_per_class_sp_100.png" alt="Alt Text" height="700">

3. In our last set of experiments, we went for exploring the possiblity of replacing ground-truth labels with pseudo-labels from the source model; following images show training purely on pseudo-labels (left) and training on the mixture of pseudo-labels and ground-truth labels with a ratio of 0.25:0.75 (right). While the training on pure pseudo-labels shows no possiblity, *R7*, *R11* and *R12* show competitive results, bringing up the opportunity to reduce the labeling budget by 25%.

    <div style="display: flex; justify-content: space-around; align-items: center;">
    <img src="experiments/notebooks/results/polar_charts/_fs_dynamic_bootstrapping_ss_None_sp_None.png" alt="First Image" height="650">
    <img src="experiments/notebooks/results/polar_charts/_fs_dynamic_bootstrapping_gt_recovery_075_ss_None_sp_None.png" alt="Second Image" height="650">
    </div>

