import math
from itertools import product
import numpy as np
import torch


class Env:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem = env_params['problem']
        self.operator = env_params['operator']
        self.problem_size = env_params['problem_size']
        self.actions, self.mask = self.init_actions_and_mask()

        self.batch_size = None
        self.solutions = None

        self.done = False
        self.patience = env_params['patience']
        self.count = 0
        self.best_avg_rewards = None


    def init_actions_and_mask(self):
        actions = []
        mask = torch.zeros([1, self.problem_size * self.problem_size], dtype=torch.bool)
        i = 0
        for subset in product(range(self.problem_size), repeat=2):
            if subset[0] == subset[1]:  # or (subset[0] == (subset[1] + 1)):
                mask[:, i] = True

            actions.append(subset)
            i += 1
        return actions, mask

    def reset(self, batch_size, test_batch=None):
        self.batch_size = batch_size
        if self.problem == 'prp':
            if test_batch is None:
                edges = torch.rand(self.batch_size, self.problem_size, self.problem_size)
                ind = np.diag_indices(edges.shape[1])
                edges[:, ind[0], ind[1]] = torch.zeros(self.problem_size)
            else:
                edges = test_batch

            nodes = torch.ones([self.batch_size, self.problem_size, 1])

        elif self.problem == 'tsp':
            instances = np.random.rand(self.batch_size, self.problem_size, 2)
            edges = np.zeros((self.batch_size, self.problem_size, self.problem_size))
            for b in range(self.batch_size):
                e = np.zeros((self.problem_size, self.problem_size))
                for i in range(self.problem_size - 1):
                    for j in range(i + 1, self.problem_size):
                        x1, y1 = instances[b, i]
                        x2, y2 = instances[b, j]
                        xdiff = x2 - x1
                        ydiff = y2 - y1
                        e[i, j] = math.sqrt(xdiff * xdiff + ydiff * ydiff)
                        e[j, i] = e[i, j]

                edges[b, :, :] = e

            edges = torch.from_numpy(edges).float().cuda()
            nodes = torch.from_numpy(instances).float().cuda()

        elif self.problem == 'gpp':
            edges = []
            connectivity = 0.5
            for b in range(self.batch_size):
                vals = torch.rand(self.problem_size * (self.problem_size - 1) // 2)  # values
                conn = torch.cuda.FloatTensor(self.problem_size * (self.problem_size - 1) // 2).uniform_() < connectivity
                vals = vals * conn
                e = torch.zeros(self.problem_size, self.problem_size)
                i, j = torch.triu_indices(self.problem_size, self.problem_size, 1)
                e[i, j] = vals
                e.T[i, j] = vals

                edges.append(e)

            edges = torch.stack(edges)
            nodes = torch.ones([self.batch_size, self.problem_size, 1])
            solutions = []
            for b in range(self.batch_size):
                perm = torch.randperm(self.problem_size)
                sol = torch.stack((perm[:self.problem_size // 2], perm[self.problem_size // 2:]))
                solutions.append(sol)

            self.solutions = torch.stack(solutions, 0)

        else:
            raise ValueError('Unsupported problem type')

        # Evaluate new batch, take reward as baseline
        bl_reward = self.get_rewards((nodes, edges))
        self.best_avg_rewards = bl_reward.mean().item()
        self.count = 0
        self.done = False

        return (nodes, edges), bl_reward, self.done

    def step(self, selected, batch, rewards=True):
        pairs = np.stack([self.actions[i] for i in selected.cpu().numpy()])
        if self.operator == 'swap':
            if self.problem == 'gpp':
                self.solution_swap(pairs)
            else:
                batch = self.swap(pairs, batch)

        elif self.operator == 'insert':
            batch = self.insert(pairs, batch)

        elif self.operator == 'reverse':
            batch = self.reverse(pairs, batch)

        else:
            raise ValueError('Unsupported operator')

        if rewards:
            reward = self.get_rewards(batch)
        else:
            reward = None

        if not self.done:
            # Stopping criteria
            mean_reward = reward.mean().item()
            if mean_reward <= self.best_avg_rewards:
                self.count += 1
                if self.count > self.patience:
                    self.done = True
            else:
                self.best_avg_rewards = mean_reward
                self.count = 0

        return batch, reward, self.done

    def swap(self, pairs, batch):
        nodes, edges = batch
        new_edges = []
        new_nodes = []
        for i in range(self.batch_size):
            order = np.arange(self.problem_size)
            order[[pairs[i, 0].item(), pairs[i, 1].item()]] = order[[pairs[i, 1].item(), pairs[i, 0].item()]]
            new_order = torch.tensor(order)
            mesh = torch.meshgrid([new_order, new_order], indexing='ij')
            new_edges.append(edges[i, :, :][mesh])
            if self.problem == 'tsp':
                new_nodes.append(nodes[i, new_order, :])

        if self.problem == 'prp':
            new_nodes = nodes
        elif self.problem == 'tsp':
            new_nodes = torch.stack(new_nodes)
        new_edges = torch.stack(new_edges)
        return new_nodes, new_edges

    def solution_swap(self, pairs):
        new_solutions = []
        for i in range(self.batch_size):
            sol = self.solutions[i, :, :]
            item_i, item_j = pairs[i, 0], pairs[i, 1]

            ind_i = (sol == item_i).nonzero()[0]
            ind_j = (sol == item_j).nonzero()[0]

            if ind_i[0] != ind_j[0]:  # they are in different clusters
                aux = sol[ind_i[0], ind_i[1]].item()
                sol[ind_i[0], ind_i[1]] = sol[ind_j[0], ind_j[1]]
                sol[ind_j[0], ind_j[1]] = aux

            new_solutions.append(sol)
        self.solutions = torch.stack(new_solutions)

    def insert(self, pairs, batch):
        nodes, edges = batch
        new_edges = []
        new_nodes = []
        for i in range(self.batch_size):
            order = np.arange(self.problem_size)

            item = pairs[i][0]
            new_order = np.delete(order, item)
            new_order = np.insert(new_order, pairs[i][1], item)

            new_order = torch.tensor(new_order)
            mesh = torch.meshgrid([new_order, new_order], indexing='ij')
            new_edges.append(edges[i, :, :][mesh])
            if self.problem == 'tsp':
                new_nodes.append(nodes[i, new_order, :])
        if self.problem == 'prp':
            new_nodes = nodes
        elif self.problem == 'tsp':
            new_nodes = torch.stack(new_nodes)
        new_edges = torch.stack(new_edges)
        return new_nodes, new_edges

    def reverse(self, pairs, batch):
        nodes, edges = batch
        new_nodes = []
        new_edges = []
        for i in range(self.batch_size):
            order = np.arange(self.problem_size)
            pos_i = pairs[i, 0].item()
            pos_j = pairs[i, 1].item()
            if pos_i < pos_j:
                sub_arr = order[pos_i:pos_j+1]
                order = np.concatenate((order[:pos_i], sub_arr[::-1], order[pos_j+1:]), axis=0)
            else:
                sub_arr = order[pos_j:pos_i+1]
                order = np.concatenate((order[:pos_j], sub_arr[::-1], order[pos_i+1:]), axis=0)

            order = torch.tensor(order)
            mesh = torch.meshgrid([order, order], indexing='ij')
            new_edges.append(edges[i, :, :][mesh])
            if self.problem == 'tsp':
                new_nodes.append(nodes[i, order, :])
        if self.problem == 'prp':
            new_nodes = nodes
        elif self.problem == 'tsp':
            new_nodes = torch.stack(new_nodes)

        new_edges = torch.stack(new_edges)

        return new_nodes, new_edges

    def get_rewards(self, batch):
        nodes, edges = batch
        if self.problem == 'prp':
            triu_indices = torch.triu_indices(self.problem_size, self.problem_size, offset=1)
            rewards = edges[:, triu_indices[0, :], triu_indices[1, :]].sum(-1)

        elif self.problem == 'tsp':
            costs = torch.zeros(self.batch_size)
            for i in range(self.problem_size - 1):
                costs += edges[:, i, i + 1]
            costs += edges[:, 0, -1]
            rewards = -costs

        elif self.problem == 'gpp':
            costs = torch.zeros(self.batch_size)
            for b in range(self.batch_size):
                cost = 0
                sol = self.solutions[b, :, :]
                e = edges[b, :, :]
                for i in range(self.problem_size // 2):
                    for j in range(self.problem_size // 2):
                        cost += e[sol[0, i], sol[1, j]]
                costs[b] = cost
            rewards = -costs

        else:
            raise ValueError('Unsupported problem type')

        return rewards


    def reorder_instance(self, instance):
        order = torch.randperm(self.problem_size)
        mesh = torch.meshgrid([order, order], indexing='ij')
        return instance[mesh].unsqueeze(0)

    def reorder_batch(self, batch):
        nodes, edges = batch
        new_nodes = []
        new_edges = []
        for i in range(self.batch_size):
            order = torch.randperm(self.problem_size)
            mesh = torch.meshgrid([order, order], indexing='ij')
            new_edges.append(edges[i, :, :][mesh])
            if self.problem == 'tsp':
                new_nodes.append(nodes[i, order, :])
        if self.problem in ['prp', 'gpp']:
            new_nodes = nodes
        elif self.problem == 'tsp':
            new_nodes = torch.stack(new_nodes)
        new_edges = torch.stack(new_edges)
        return new_nodes, new_edges
