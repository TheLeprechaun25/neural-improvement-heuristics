import os
from Trainer import Trainer
from options.train_options import parse_train_args


main_dir = os.path.dirname(os.path.realpath(__file__)) + '/'


def main():
    env_params, model_params, optimizer_params, trainer_params = parse_train_args(main_dir)
    trainer = Trainer(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)
    trainer.run()


if __name__ == "__main__":
    main()
