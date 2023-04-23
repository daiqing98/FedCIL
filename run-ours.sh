# FedCIL 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Mnist-cl-5-EMnist-B --algorithm FedCL --batch_size 32 --gen_batch_size 32  --num_glob_iters 200 --local_epochs 400 --num_users 5 --times 3 --device "cuda" --offline 0 --lr-CIFAR 0.0001 --naive 0 --TASKS 5 --ft_iters=0 --fedavg=0 --md_iter 50

# FedCIL 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Mnist-cl-5-EMnist-B --algorithm FedCL --batch_size 32 --gen_batch_size 32  --num_glob_iters 200 --local_epochs 400 --num_users 5 --times 3 --device "cuda" --offline 0 --lr-CIFAR 0.0001 --naive 0 --TASKS 5 --ft_iters=0 --fedavg=0 --md_iter 100

# ACGAN + FedAvg
CUDA_VISIBLE_DEVICES=2 screen python main.py --dataset Mnist-cl-5-EMnist-B --algorithm FedCL --batch_size 32 --gen_batch_size 32  --num_glob_iters 200 --local_epochs 400 --num_users 5 --times 3 --device "cuda" --offline 0 --lr-CIFAR 0.0001 --naive 1 --TASKS 5 --ft_iters=0 --fedavg=0 --md_iter 0
