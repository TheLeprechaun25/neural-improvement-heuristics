# Neural Improvement Heuristics for Graph Combinatorial Optimization Problems

Official implementation of the paper "Neural Improvement Heuristics for Graph Combinatorial Optimization Problems" (https://arxiv.org/abs/2206.00383). Published in *IEEE Transactions on Neural Networks and Learning Systems*.

## Introduction
This paper presents a Neural Improvement (NI) method to solve graph combinatorial optimization problems. 
NIH is a general framework that can be applied to various graph combinatorial optimization problems. 
In the paper we primarily focus on the Preference Ranking Problem (PRP) but we also evaluate NI on the Traveling Salesman Problem (TSP), and the Graph Partitioning Problem (GPP).

The model consists of an anisotropic Graph Neural Network that takes as input node and edge features and outputs node and edge embeddings.
The edge embeddings are then used to compute a probability distribution over the edges of the graph.
Finally, an action (edge) is sampled from this distribution and the graph is updated accordingly with the use of an operator that modifies the graph structure.

## Requirements

* Python 3.8
* NumPy
* PyTorch 

## Usage

### Training

To train the model, run the following command:

```bash
python train.py --problem <problem_name>
```

For example, to train the model on the *PRP* problem with the *insert* operator for *100 epochs*, run the following command:

```bash
python train.py --problem prp --operator insert --epochs 100
```

To load a pre-trained model, run the following command:

```bash
python train.py --problem <problem_name> --model_load_path <model_path>
```

See `options/train_options.py` for more options and default hyperparameters.

### Testing

To test the model, run the following command:

```bash
python inference.py --problem <problem_name> --operator <operator_name> --model_load_path <model_path>
```

See `options/test_options.py` for more options and default hyperparameters.

## Cite

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{garmendia2023neural,
  title={Neural Improvement Heuristics for Graph Combinatorial Optimization Problems},
  author={Garmendia, Andoni I and Ceberio, Josu and Mendiburu, Alexander},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement

This repository has been based on the following repositories:
* [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://github.com/yd-kwon/POMO)

## License

MIT License




