## Insruction

1. Go to ./FLAlgorithms/generator and replace the original model.py and gan.py with scripts in this folder (remember to keep a copy)
2. Run this command

```python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Mnist-cl-5-EMnist-B --algorithm FedCL --batch_size 32 --gen_batch_size 32  --num_glob_iters 200 --local_epochs 400 --num_users 5 --times 3 --device "cuda" --offline 0 --lr-CIFAR 0.0001 --naive 1 --TASKS 5 --ft_iters=0 --fedavg=0 --md_iter 0
```