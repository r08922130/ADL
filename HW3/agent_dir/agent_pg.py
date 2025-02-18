import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from agent_dir.agent import Agent
from environment import Environment
from logger import Logger
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('best/pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.action_probs = torch.tensor([])

        #logger
        self.logger = Logger('pg')

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []
        self.action_probs = torch.tensor([])

    def make_action(self, state, test=False):
        #action = self.env.action_space.sample() # TODO: Replace this line!
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        #print(torch.tensor(state).unsqueeze(0))
        
        prob = self.model(torch.tensor(state).unsqueeze(0))
        if not test:
            cate = Categorical(prob)
            action = cate.sample()
            self.action_probs = torch.cat((self.action_probs, cate.log_prob(action)))
        else:
            action = prob.topk(1)[1][0]
            
        #print(cate.log_prob(action)[0])
        
        
        #print(self.action_probs)
        #print(action)
        return action.numpy()[0]

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        
        num_r = len(self.rewards)
        #print(self.rewards)
        R = [0] * (num_r)
        R[0] = self.rewards[-1]
        for i in range(num_r-1):
            R[i+1] = self.rewards[-(i+2)] + self.gamma* R[i]
        
        
        R.reverse()
        #print(R)
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        R = torch.FloatTensor(R)
        #R = (R-R.mean())/(R.std()+np.finfo(np.float32).eps)

        #variance reduction
        R = (R-R.mean())
        loss = torch.sum(-R* self.action_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            self.logger.write('Epochs: %d/%d | Avg reward: %f \n'%
                    (epoch, self.num_episodes, avg_reward))

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
