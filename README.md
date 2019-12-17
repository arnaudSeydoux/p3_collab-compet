[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"




## Introduction

This project trains a reinforcement learning agent to play a game.

We use for the project the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Getting Started

1. You should have python 3.7 installed on your machine.

2. Clone the repo into a local directory

    Then cd in it:
    `cd p2-continuous-control`


3. To install the dependencies, we advise you to create an environment.
    If you use conda, juste run:
    `conda create --name p2cc python=3.7`
    to create `dqn` environment

    Then install requirements files:
    `pip install -r requirements.txt`

4. Run the training file to train the agent, and see progress in training along time
`python train.py`

5. You'll find more infos on the project in `Report.md` file.




