# PROSE
This software project accompanies the research paper, ["PROSE: Preference Reasoning by Observing and Synthesizing Examples"](https://arxiv.org/pdf/2505.23815).

We create personalizable agents by splitting an interactive task into two stages: (1) identifying the preferences and then (2) conditioning agents on these preferences. 

This project focuses on the first stage and answers the following research questions:

- *RQ1*: Does providing adversarial comparison trajectories lead to more accurate and precise preferences? 
- *RQ2*: Does breaking down preferences into components lead to faster and more generalizable inference?
- *RQ3*: Does confirming a preference across multiple samples lead to more accurate and precise preferences?

## Installation
1. Create a Python 3.9+ environment with `venv` or `conda`, and then activate it
2. Clone the repo and `cp` into `ml-predict`
3. Make a copy of `preference_inferrer/common/personal_keys_template.py` and name it 
`preference_inferrer/common/personal_keys.py`:
```
cp preference_inferrer/common/personal_keys_template.py preference_inferrer/common/personal_keys.py
```
4. Fill in the required keys in `preference_inferrer/common/personal_keys.py`. This is necessary to set up OpenAI, HuggingFace, and WandB access.
5. Install correct pytorch for machine (see: https://pytorch.org/get-started/locally/)
6. Install this repository: `pip install .`


## PLUME: Preference Learning from User Memos and Emails
This repository contains an assistive writing task. All code related to it can be 
found in `preference_inferrer/tasks`

To replicate the results from the paper, the seeds used were {1235, 9328, 235, 6936, 2045}.

The objective of this framework is to write a summary or an email (depending on the sub-task) on behalf of the user
that complies with the user's preferences.

## Running experiments
All experimental configs are found in parameters section of `experiment_config.yaml`.  
- To run a single experiment locally, input the desired configurations in `experiment_config.yaml` and run `python scripts/run_single_experiment.py`
- To run a set of experiments with different configurations, run: `python scripts/run_all_experiments.py`

## Viewing prompts
You can view what a full sequence of prompts looks like by running:  
`python -m preference_inferrer.llms.test_llm`  
You can configure either the `experiment_config.yaml` file or the parameters directly within `test_llm.py` to change the prompt 
configurations.

## PRELUDE Attribution.
PLUME is derived from PRELUDE.
Part of the PLUME code is derived from: 
https://github.com/gao-g/prelude/tree/246b51c8b381a3a09ef0e231c1e60605f15b804a

## Citations
If you use this repository, please cite:

```
@article{aroca2024prose,
  title={Aligning LLMs by Predicting Preferences from User Writing Samples},
  author={Aroca-Ouellette, Stephane and Mackraz, Natalie and Theobald, Barry-John and Metcalf, Katherine},
  journal={ICML},
  year={2025}
}
```

Please also consider citing PRELUDE:
```
@article{gao2024aligning,
  title={Aligning llm agents by learning latent preference from user edits},
  author={Gao, Ge and Taymanov, Alexey and Salinas, Eduardo and Mineiro, Paul and Misra, Dipendra},
  journal={NeurIPS},
  year={2024}
}
```
