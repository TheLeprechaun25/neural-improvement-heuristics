import argparse
from utils import generate_word


def parse_train_args(main_dir):
    parser = argparse.ArgumentParser(description='Train a neural improvement heuristic on a specific problem.')

    # Environment
    parser.add_argument('--problem', type=str, default='prp', choices=['prp', 'tsp', 'gpp'], help='problem to solve (prp, tsp, gpp)')
    parser.add_argument('--operator', type=str, default='insert', help='operator to use (insert, swap, reverse)')
    parser.add_argument('--problem_size', type=int, default=20, help='problem size')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='number of GNN encoder layers')
    parser.add_argument('--aggregation', type=str, default='sum', choices=['sum', 'mean', 'max'], help='aggregation function')
    parser.add_argument('--graph_aggregation', type=str, default='mean', choices=['sum', 'mean', 'max'], help='graph aggregation function')
    parser.add_argument('--normalization', type=str, default='batch', choices=['batch', 'instance', 'none'], help='normalization function')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--tanh_clipping', type=float, default=10., help='tanh clipping value')

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--milestones', type=int, nargs='+', default=[501, ], help='milestones for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma for learning rate scheduler')

    # Training
    parser.add_argument('--train_steps', type=int, default=4, help='number of training steps')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--n_episodes', type=int, default=10, help='number of episodes per epoch')
    parser.add_argument('--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--disc_reward_gamma', type=float, default=0.9, help='discount factor for discriminator rewards')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm')

    # Test
    parser.add_argument('--test_steps', type=int, default=100, help='number of test steps')
    parser.add_argument('--test_batch_size', type=int, default=128, help='batch size for testing')
    parser.add_argument('--test_frequency', type=int, default=1, help='test frequency')

    # Others
    parser.add_argument('--model_load_path', type=str, default='', help='path to pretrained model')
    parser.add_argument('--no_verbose', action='store_true', help='print results')

    args = parser.parse_args()

    env_params = {
        'problem': args.problem,
        'operator': args.operator,
        'problem_size': args.problem_size,
        'patience': args.patience,
    }

    model_params = {
        'embedding_dim': args.embedding_dim,
        'n_encode_layers': args.n_encode_layers,
        'aggregation': args.aggregation,
        'graph_aggregation': args.graph_aggregation,
        'normalization': args.normalization,
        'n_heads': args.n_heads,
        'tanh_clipping': args.tanh_clipping,
    }

    optimizer_params = {
        'optimizer': {
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        'scheduler': {
            'milestones': args.milestones,
            'gamma': args.gamma
        }
    }

    trainer_params = {
        # Training
        'train_steps': args.train_steps,
        'epochs': args.epochs,
        'n_episodes': args.n_episodes,
        'train_batch_size': args.train_batch_size,
        'disc_reward_gamma': args.disc_reward_gamma,
        'max_grad_norm': args.max_grad_norm,

        # Test
        'test_steps': args.test_steps,
        'test_batch_size': args.test_batch_size,
        'test_frequency': args.test_frequency,

        # Others
        'main_dir': main_dir,
        'model_load': {
            'enable': False if args.model_load_path == '' else True,
            'path': args.model_load_path
        },
        'verbose': not args.no_verbose,
    }

    if env_params['problem'] == 'prp':
        model_params['node_dim'] = 1
        model_params['edge_dim'] = 2
    elif env_params['problem'] == 'tsp':
        model_params['node_dim'] = 2
        model_params['edge_dim'] = 3
    elif env_params['problem'] == 'gpp':
        assert env_params['operator'] == 'swap', "Only swap operator is supported for GPP"
        model_params['node_dim'] = 1
        model_params['edge_dim'] = 2

    execution_name = generate_word(6)
    trainer_params['execution_name'] = execution_name


    return env_params, model_params, optimizer_params, trainer_params
