from unityagents import UnityEnvironment
from ddpg_agent import Agent
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from noises import OUNoise, GaussianNoise
import json

def ddpg(agent, simul_name, best_score, n_episodes=1000, max_t=200, print_every=100):
    # print(f'{num_agents} agents')
                     # Only save the agent if he gets a result better than 30.0
    # Number of episodes needed to solve the environment (mean score of 30 on the 100 last episodes)
    episode_solved = n_episodes
    scores_max_agent = []             # list containing mean scores (over the 20 agents) from each episode
    scores_mean_last100 = []           # List containing mean value (over the 20 agents) of score_window
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents) 

        states = env_info.vector_observations
#         print(f'state: {state} {len(state)}')
#         agent.reset()
        actions = []
        for t in range(max_t):
            actions = agent.act(states)
            env_info= env.step(actions)[brain_name]
            
            next_states = env_info.vector_observations       # get next state (for each agent)
#             print(f'next_states: {next_states}')
            rewards = env_info.rewards                        # get reward (for each agent)
            dones = env_info.local_done 
            
            agent.step(states, actions, rewards, next_states, dones)
            # print(f'states: {states}')
            # print(f'actions: {actions}')
            # print(f'rewards: {rewards}')
            # print(f'next_states: {next_states}')
            # print(f'dones: {dones}')

            states = next_states
            scores += rewards
            if np.any(dones):
                break
        # agent.reset()
        agent.end_episode()


        scores_window.append(scores.max())                # save most recent max score between agent
        scores_max_agent.append(scores.max())             # save most recent max score between agent
        scores_mean_last100.append(np.mean(scores_window))
        if (i_episode % print_every) == 0:
            print(f'Episode {i_episode} Sc: {scores.max():.2f} Av {np.mean(scores_window):.2f} a {actions[0]} {actions[1]} t {t}')
        # print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, scores.mean()), end="")
        if(np.mean(scores_window)>best_score):
            break
    torch.save(agent.actor_local.state_dict(), f'{simul_name}_actor.pth')
    torch.save(agent.critic_local.state_dict(), f'{simul_name}_agent.pth')

    with open(f'{simul_name}.json', 'w') as filehandle:
        json.dump(scores_max_agent, filehandle)
        
    return scores_max_agent

random_seed = mu = theta = sigma = learnings = learning_rates = factor = doClip = network = None


def reset_variables():
    global best_score, n_episodes, add_noise, num_agents, random_seed , tau_params, mu , theta , sigma , learnings , learning_rates , factor , doClip , network

    tau_params = (0.01, 0.001, 0.999)
    random_seed = 0
    mu = 0.0
    # theta= .15
    theta= .3
    sigma = .4
    # sigma = 2

    # learnings = (5,5)
    learnings = (10,10)
    # learning_rates = (1e-3, 1e-4)
    learning_rates = (1e-4, 1e-4)

    factor = .6
    doClip = True
    add_noise = False
    network = (256, 512)

    num_agents = 2

    # used to improve hyper parameters, we stop at 400 steps
    n_episodes=3000
    # best_score = .05
    best_score = .5

def runSimul():
    # print(f'num_agents {num_agents}')
    noise = [OUNoise(brain.vector_action_space_size, random_seed,mu=mu, theta=theta, sigma=sigma) for _ in range(num_agents)]
    simul_name = f'./results/agents={num_agents},states={states},a_s={brain.vector_action_space_size},seed={random_seed},taus={tau_params},noise={None},learns={learnings},learns_r={learning_rates},net={network},doClip={doClip},noise={add_noise},t={n_episodes},maxScr={best_score}'
    print(simul_name)

    # noise = GaussianNoise(brain.vector_action_space_size, factor)
    agent = Agent(num_agents = num_agents, state_size=states, action_size=brain.vector_action_space_size, random_seed=random_seed, tau_params = tau_params, noise=noise, learnings = learnings, learning_rates = learning_rates, network = network, doClip = doClip,  add_noise = add_noise)
    # scores = ddpg(n_episodes=1000, max_t=300, print_every=100)
    scores= ddpg(agent, simul_name, best_score = best_score, n_episodes=n_episodes, max_t=100, print_every=100)

    with open(f'{simul_name}.json', 'w') as filehandle:
                    json.dump(scores, filehandle)     


env = UnityEnvironment(file_name='Tennis.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

states = env_info.vector_observations.shape[1]

reset_variables()
print(f'network: {network}')

runSimul()
# network = (128, 64)
# network = (512, 256)


# decay_rate = .
# for steps in [1,5,10,50]:
#     for times in [1,5,10,50]:
#         filename = f'./results/steps={steps}, times={times}'
#         print(filename)
#         noise = OUNoise(brain.vector_action_space_size, random_seed,mu=mu, theta=theta, sigma=sigma)
#         agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size, random_seed=random_seed, noise=noise, learnings = (steps,times),network = (256, 128), doClip = doClip)
#         # scores = ddpg(n_episodes=1000, max_t=300, print_every=100)
#         scores, _ = ddpg(n_episodes=100, max_t=10000, print_every=100)

#         with open(f'{filename}.json', 'w') as filehandle:
#                         json.dump(scores, filehandle)     





for learning_rates in [(1e-3, 1e-4), (1e-4, 1e-3),(1e-3, 1e-3),(1e-4, 1e-4),(1e-5, 1e-5)]:
    runSimul()
reset_variables()



# for doClip in [True, False]:
#     runSimul()
# reset_variables()

for learnings in [(1, 1),(5, 5),(10, 10),(20, 20), (20, 10), (10,20)]:
    runSimul()
reset_variables()


for network in [(256*2,512*2),(64, 128),(256, 512)]:
    runSimul()
reset_variables()

for tau_params in [(0.01, 0.001, 0.999),(0.001, 0.001, 0.999),(0.1, 0.001, 0.99)]:
    runSimul()

# for sigma in [1., .8, .5, .2, .1, .01]:
#     runSimul()
# reset_variables()
