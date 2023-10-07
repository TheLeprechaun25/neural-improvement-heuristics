import argparse


def parse_test_args(main_dir):
    parser = argparse.ArgumentParser(description='Test a neural improvement heuristic.')

    # Environment
    parser.add_argument('--problem', type=str, default='prp', choices=['prp', 'tsp', 'gpp'], help='problem to solve (prp, tsp, gpp)')
    parser.add_argument('--operator', type=str, default='insert', help='operator to use (insert, swap, reverse)')
    parser.add_argument('--problem_size', type=int, default=20, help='problem size')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='number of GNN encoder layers')
    parser.add_argument('--aggregation', type=str, default='sum', choices=['sum', 'mean', 'max'], help='aggregation function')
    parser.add_argument('--graph_aggregation', type=str, default='mean', choices=['sum', 'mean', 'max'], help='graph aggregation function')
    parser.add_argument('--normalization', type=str, default='batch', choices=['batch', 'instance', 'none'], help='normalization function')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--tanh_clipping', type=float, default=10., help='tanh clipping value')

    # Test
    parser.add_argument('--n_episodes', type=int, default=10, help='number of different episodes (initializations)')
    parser.add_argument('--test_steps', type=int, default=100, help='number of test steps')
    parser.add_argument('--test_batch_size', type=int, default=128, help='batch size for testing')
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

    tester_params = {
        # Test
        'n_episodes': args.n_episodes,
        'test_steps': args.test_steps,
        'test_batch_size': args.test_batch_size,

        # Others
        'main_dir': main_dir,
        'model_load_path': args.model_load_path,
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

    return env_params, model_params, tester_params
