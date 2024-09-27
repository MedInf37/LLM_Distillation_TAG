# LLM Distillation on Text-Attributed Graphs (TAGs)

This repository contains code for performing distillation of large language models (LLMs) on text-attributed graph (TAG) datasets.

## Important Notes

- You may need to modify the file structure, as the code is currently configured for macOS and Linux environments with a Condor job scheduling system.

## Usage Instructions

- To run the experiments, execute the following commands:

`python experiments.py --method train_lm --config config/train_lm_config.yaml`

`python experiments.py --method train_graph --config config/train_graph_config.yaml`

`python experiments.py --method tune_hyperparameters --config config/tune_hyperparameters_config.yaml`

`python experiments.py --method tune_hyperparameters --config config/tune_hyperparameters_graph_config.yaml`

`python experiments.py --method distill_experiment`

`python experiments.py --method save_embedding --config config/save_embedding_config.yaml`

`python experiments.py --method distill_baseline --config config/distill_baseline_config.yaml`


## Acknowledgements
This project utilizes code from the DeepGCNs repository:
- Repository: https://github.com/lightaime/deep_gcns_torch.git
- Paper: "DeepGCNs: Can GCNs Go as Deep as CNNs?" by Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem (2019)
- Paper URL: https://arxiv.org/abs/1904.03751