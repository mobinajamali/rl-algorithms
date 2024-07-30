from agent import Agent
from utils import plot_learning_curve
import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(lr_critic=0.0001, lr_actor=0.001, input_dims=env.observation_space.shape,
                  tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=env.action_space.shape[0])
    n_games = 900
    filename = 'LunarLander_lr_critic' + str(agent.lr_critic) + 'LunarLander_lr_actor' + str(agent.lr_actor) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation, _ = env.reset()
        done, trunc = False, False
        score = 0
        agent.noise.reset()
        while not (done or trunc):
            action = agent.choose_action(observation)
            observation_, reward, done, trunc, info = env.step(action)
            terminal = done or trunc
            agent.store_transition(observation, action, reward, observation_, terminal)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()
        
        print(f'episode {i} score = {score}')
    
    # generate a list of episode numbers
    x =[i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)