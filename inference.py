import os
from Tester import Tester
from options.test_options import parse_test_args


main_dir = os.path.dirname(os.path.realpath(__file__)) + '/'


def run_inference():
    env_params, model_params, tester_params = parse_test_args(main_dir)
    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    tester.run()


if __name__ == "__main__":
    run_inference()
