import torch
from torch.optim import AdamW as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from networks.model import Model
from Env import Env
from utils import clip_grad_norms


class Trainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type(torch.FloatTensor)

        self.env = Env(**self.env_params)

        self.model = Model(**self.model_params).to(device)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        model_load = trainer_params['model_load']
        if model_load['enable']:
            path = model_load['path']
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            if 'optimizer_state_dict' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        try:
            test_batch = torch.load(f"{self.trainer_params['main_dir']}data/{self.env.problem}/n_{self.env.problem_size}/n_{self.env.problem_size}.pt")
            self.test_batch = test_batch[:self.trainer_params['test_batch_size']].to(device)
            baseline_results = torch.load(f"{self.trainer_params['main_dir']}data/{self.env.problem}/n_{self.env.problem_size}/results_n_{self.env.problem_size}.pt")
            self.best_known_values = torch.tensor(baseline_results['ma'][:self.trainer_params['test_batch_size']])
        except:
            print("No test batch found. Generating new one...")
            (node, edge), _, _ = self.env.reset(self.trainer_params['test_batch_size'])
            self.test_batch = edge
            self.best_known_values = None

    def run(self):
        for epoch in range(1, self.trainer_params['epochs']+1):
            print(f"\nEpoch {epoch}/{self.trainer_params['epochs']})")

            # Train
            for ep in range(self.trainer_params['n_episodes']):
                r, avg_loss = self.train_one_episode()
                if self.trainer_params['verbose']:
                    print(f"Epoch {epoch} - Episode {ep}/{self.trainer_params['n_episodes']}) Best rewards: {r:.3f}, Avg loss: {avg_loss:.3f}")

            # Evaluate and save
            if epoch % self.trainer_params['test_frequency'] == 0:
                eval_rewards = self.evaluate()
                if self.trainer_params['verbose']:
                    print(f"Eval rewards: {eval_rewards:.3f}")
                path = f"{self.trainer_params['main_dir']}saved_models/{self.env.problem}/{self.trainer_params['execution_name']}_{self.env.operator}_epoch{epoch}_n{self.env.problem_size}_{eval_rewards:.2f}.pt"
                if isinstance(self.model, torch.nn.DataParallel):
                    model_state_dict = self.model.module.state_dict()
                else:
                    model_state_dict = self.model.state_dict()
                torch.save({
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)

    def train_one_episode(self):
        # Generate new batch
        batch, bl_reward, done = self.env.reset(self.trainer_params['train_batch_size'])

        avg_loss = 0
        t = 0
        while not done:
            t += 1
            log_probs = []
            rewards = []
            self.optimizer.zero_grad()
            for step in range(self.trainer_params['train_steps']):
                # Get action probabilities
                log_p = self.model(batch, self.env.mask, self.env.problem, self.env.solutions)
                probs = log_p.exp()
                selected = probs.multinomial(1).squeeze(1)
                log_p = log_p.gather(1, selected.unsqueeze(-1)).squeeze(-1)
                log_probs.append(log_p)

                # Perform the action
                batch, reward, done = self.env.step(selected, batch)
                rewards.append(reward - bl_reward)

                # New baseline
                bl_reward = reward

            # Calculate discounting rewards
            t_steps = torch.arange(len(rewards))
            discounts = self.trainer_params['disc_reward_gamma'] ** t_steps
            r = [r_i * d_i for r_i, d_i in zip(rewards, discounts)]
            r = r[::-1]
            b = torch.cumsum(torch.stack(r), dim=0)
            c = [b[k, :] for k in reversed(range(b.shape[0]))]
            R = [c_i / d_i for c_i, d_i in zip(c, discounts)]

            # Compute loss and back-propagate
            policy_loss = []
            for log_prob_i, r_i in zip(log_probs, R):
                policy_loss.append(-log_prob_i * r_i)
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            _ = clip_grad_norms(self.optimizer.param_groups, self.trainer_params['max_grad_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            avg_loss += loss.item()

        avg_loss /= t

        return self.env.best_avg_rewards, avg_loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        if self.trainer_params['verbose']:
            print(f"\nEvaluating model...")

        # Generate new batch
        batch, best_rewards, _ = self.env.reset(self.trainer_params['test_batch_size'], self.test_batch)

        # Evaluate
        for step in range(self.trainer_params['test_steps']):
            log_p = self.model(batch, self.env.mask, self.env.problem, self.env.solutions)
            probs = log_p.exp()

            # Sample from probabilities
            selected = probs.multinomial(1).squeeze(1)

            # Perform the action
            batch, rewards, _ = self.env.step(selected, batch)
            best_rewards = torch.max(best_rewards, rewards)

            if (step % 10 == 0 or step == self.trainer_params['test_steps'] - 1) and self.trainer_params['verbose']:
                if self.best_known_values is not None:
                    gap = (100 * (self.best_known_values - rewards) / self.best_known_values).mean().item()
                    best_gap = (100 * (self.best_known_values - best_rewards) / self.best_known_values).mean().item()
                    print(f"Step {step+1}/{self.trainer_params['test_steps']}) Gap: {gap:.2f}%, Best gap: {best_gap:.2f}%")
                else:
                    print(f"Step {step+1}/{self.trainer_params['test_steps']}) Rewards: {rewards.mean().item():.2f}, Best rewards: {best_rewards.mean().item():.2f}")
        return best_rewards.mean().item()

