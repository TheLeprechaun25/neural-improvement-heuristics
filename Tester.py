import torch
from networks.model import Model
from Env import Env


class Tester:
    def __init__(self, env_params, model_params, tester_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type(torch.FloatTensor)

        # Initialize environment and model
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore model weights
        try:
            model_state_dict = torch.load(tester_params['model_load_path'])['model_state_dict']
        except FileNotFoundError:
            print("No model found in the specified path. Specify a valid path with --model_load_path.")
            exit()

        self.model.load_state_dict(model_state_dict)
        self.model.to(device)

        try:
            test_batch = torch.load(f"{self.tester_params['main_dir']}data/{self.env.problem}/n_{self.env.problem_size}/n_{self.env.problem_size}.pt")
            self.test_batch = test_batch[:self.tester_params['test_batch_size']].to(device)
            baseline_results = torch.load(f"{self.tester_params['main_dir']}data/{self.env.problem}/n_{self.env.problem_size}/results_n_{self.env.problem_size}.pt")
            self.best_known_values = torch.tensor(baseline_results['ma'][:self.tester_params['test_batch_size']])
        except:
            print("No test batch found. Generating new one...")
            (node, edge), _, _ = self.env.reset(self.tester_params['test_batch_size'])
            self.test_batch = edge
            self.best_known_values = None


    def run(self):
        overall_best_rewards = torch.ones(self.tester_params['test_batch_size']) * -float('inf')
        for episode in range(1, self.tester_params['n_episodes']+1):
            print(f"\nEpisode {episode}/{self.tester_params['n_episodes']})")
            batch, _, _ = self.env.reset(self.tester_params['test_batch_size'], self.test_batch)
            batch = self.env.reorder_batch(batch)
            rewards = self.run_episode(batch)
            overall_best_rewards = torch.max(overall_best_rewards, rewards)

        if self.best_known_values is not None:
            gap = (100 * (self.best_known_values - overall_best_rewards) / self.best_known_values).mean().item()
            print(f"\nOverall best rewards: {overall_best_rewards.mean().item():.2f} (gap: {gap:.2f}%)")
        else:
            print(f"\nOverall best rewards: {overall_best_rewards.mean().item():.2f}")

    def run_episode(self, batch):
        self.model.eval()

        best_rewards = self.env.get_rewards(batch)

        for step in range(self.tester_params['test_steps']):
            log_p = self.model(batch, self.env.mask, self.env.problem, self.env.solutions)
            probs = log_p.exp()

            # Sample from probabilities
            selected = probs.multinomial(1).squeeze(1)

            # Perform the action
            batch, rewards, done = self.env.step(selected, batch)
            best_rewards = torch.max(best_rewards, rewards)

            if ((step % 10 == 0) or (step == self.tester_params['test_steps'] - 1) or done) and self.tester_params['verbose']:
                if self.best_known_values is not None:
                    gap = (100 * (self.best_known_values - rewards) / self.best_known_values).mean().item()
                    best_gap = (100 * (self.best_known_values - best_rewards) / self.best_known_values).mean().item()
                    print(f"Step {step+1}, Cur gap: {gap:.2f}%, Best found gap: {best_gap:.2f}%")
                else:
                    print(f"Step {step+1}, Rewards: {rewards.mean().item():.2f}, Best rewards: {best_rewards.mean().item():.2f}")

            if done:
                print(f"Patience reached. Episode finished after {step+1} steps.")
                break

        return best_rewards
