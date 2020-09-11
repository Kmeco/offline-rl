# Offline Reinforcement Learning

**[Overview](#overview)** | **[Installation](#installation)** | **[Agents](#agents)** | **[Examples](#examples)** |

![PyPI Python Version](https://img.shields.io/pypi/pyversions/dm-acme)
![PyPI version](https://badge.fury.io/py/dm-acme.svg)

### Overview

This repository is a part of my master thesis project at UCL. 
It builds upon the [acme](https://github.com/deepmind/acme) framework and implements 
two new offline RL algorithms.

The experiments here are run on the [MiniGrid](https://github.com/maximecb/gym-minigrid) 
environemnt, but the code is modular and a new environemnt can be tested simply by 
implementing a new `_build_environment()` func that returns an environment in 
appropriate wrappers. 


### Installation

An example of a working environment is set up in each of the example 
colaboratory notebooks provided.

### Agents

This repo implements 3 different algorithms: 
* Conservative Q-learning ([CQL](https://arxiv.org/abs/2006.04779))
* Critic Regularized Regression ([CRR](https://arxiv.org/abs/2006.15134))
* Behavioural Cloning adopted from acme

### Examples

after setting up a wandb account, all the results of our experiments along with the 
versioned datasets can be accessed [here](https://app.wandb.ai/kmeco/offline-rl)

New datasets can be easily collected using the [dataset_collection_pipeline](https://colab.research.google.com/github/Kmeco/offline-rl/blob/master/dataset_collection_pipeline.ipynb) colab notebook.

Experiments can be run from [run_experiment_pipeline](https://colab.research.google.com/github/Kmeco/offline-rl/blob/master/colab_pipelines/run_experiment_pipeline.ipynb) notebook. 

Both of these notebooks are well documented. Each new experiment that is run is tracked and checkpointed to WandB. 
If you'd like to resume an existing run, it is sufficient to pass the specific `run_id`  as a '--wandb_id' flag to any of the algorithm run scripts. 



