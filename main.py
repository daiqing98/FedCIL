#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serverpFedCL import FedCL

from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool

import random

def create_server_n_user(args, i):
    
    # create base model, irreverent to FedXXX
    model = create_model(args.model, args.dataset, args.algorithm)
    
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, i)
    elif ('FedCL' in args.algorithm):
        server = FedCL(args, model, i)
    elif ('FedLwF' in args.algorithm):
        server = FedLwF(args, model, i)
    elif ('FedGR' in args.algorithm):
        server = FedGR(args, model, i)
    
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i, seed):
    
#     seed = random.randint(0,100)
#     seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    print('random seed is: ', seed)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        #server.test()

def main(args):
    seed = [9,8,7,6,5,4,3,2,1,0]
    for i in range(args.times):
        run_job(args, i, seed[i])
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    
    # AC- GAN:
    parser.add_argument('--generator-z-size', type=int, default=110)
    parser.add_argument('--generator-c-channel-size', type=int, default=64)
    parser.add_argument('--generator-g-channel-size', type=int, default=64)
    
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--generator-lambda', type=float, default=10.)
    parser.add_argument('--weight-decay', type=float, default=1e-05)
    parser.add_argument('--lr', type=float, default=1e-04)   
    parser.add_argument('--only-critic', action='store_true' )
    parser.add_argument('--offline', type=int, default=0 )
    parser.add_argument('--naive', type=int, default=0 )
    
    # CIFAR10:
    parser.add_argument('--lr-CIFAR', type=float, default=0.0002 )
    parser.add_argument('--TASKS', type=int, default=5 )
    parser.add_argument('--fedavg', type=int, default=0 )
    parser.add_argument('--ft_iters', type=int, default=0 )
    parser.add_argument('--fedcl', type=int, default=1 )
    parser.add_argument('--fedlwf', type=int, default=0 )
    parser.add_argument('--md_iter', type=int, default=100 )
    
    # GR
    parser.add_argument('--iter_GR', type=int, default=2000 )
    
    # tools
    parser.add_argument('--seed', type=int, default=1 )
    parser.add_argument('--visual', type=int, default=0 )
    parser.add_argument('--draw', type=int, default=0 )
    
    args = parser.parse_args() 

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)