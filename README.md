<<<<<<< HEAD
# Darija-LLM-Benchmark  
=======
# CodeSwitch-Darija-NLP
>>>>>>> 8c8fad538e5c7a171c628ceef6496ed5bb187197
# CodeSwitch-Darija-NLP: Benchmarking Transformers & LLMs on Moroccan Darija

## Overview
This research repository benchmarks classical Machine Learning baselines, encoder-only Arabic/Darija Transformers (**AraBERT**, **MARBERTv2**, **DarijaBERT**), and Parameter-Efficient Fine-Tuned Large Language Models (**Qwen-2.5** via QLoRA) on Moroccan Darija text classification and code-switching data.

## Project Architecture
- `data/`: Raw and preprocessed Moroccan Darija datasets.
- `notebooks/`: Step-by-step reproducible experiments (EDA, Baselines, BERT Fine-Tuning, QLoRA LLMs).
- `src/`: Modular text normalization (`preprocessing.py`) and unified evaluation pipeline (`metrics.py`).
- `results/`: Automated benchmark tables (`benchmark_table.csv`) and confusion matrices.
- `paper/`: LaTeX source files for the IEEE-formatted research report.

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn datasets transformers peft torch matplotlib seaborn`
2. Run `notebooks/01_EDA_and_Cleaning.ipynb` to clean and split the dataset.
3. Run `notebooks/02_Baseline_TFIDF_ML.ipynb` to generate baseline metrics.
4. Run `notebooks/03_FineTune_Transformers.ipynb` and `04_QLoRA_Qwen_Llama.ipynb` on GPU.