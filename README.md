# Offline Reinforcement Learning

**[Overview](#overview)** | **[Installation](#installation)** | **[Agents](#agents)** | **[Examples]** |

![PyPI Python Version](https://img.shields.io/pypi/pyversions/dm-acme)
![PyPI version](https://badge.fury.io/py/dm-acme.svg)

### Overview

This repository is a part of my master thesis project at UCL. It builds upon the [acme](https://github.com/deepmind/acme) framework and implements two new offline RL algorithms.

The experiments here are run on the [MiniGrid](https://github.com/maximecb/gym-minigrid) environemnt, but the code is modular and a new environemnt can be tested simply by implementing a new `_build_environment()` func that returns an environment in appropriate wrappers. 


### Installation

An example of a working environment is set up in each of the example colaboratory notebooks provided.

### Agents

This repo implements 3 different algorithms: 
* Conservative Q-learning ([CQL](https://arxiv.org/abs/2006.04779))
* Critic Regularized Regression ([CRR](https://arxiv.org/abs/2006.15134))
* Behavioural Cloning adopted from acme

### Examples



