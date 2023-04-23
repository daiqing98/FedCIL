# FedCIL

 Implementation of *Better Generative Replay for Continual Federated Learning, ICLR' 23*



## Usage

1. Install some of required packages. You can also install them manually.

```
pip install -r requirements.txt # install requirements
```

2. Download the processed dataset here: [Google Drive](https://drive.google.com/file/d/1F7li0NbFWbdaMsqpGUGevEYbT8TAsAx3/view?usp=share_link).
   Unzip this file and place it in the root directory.

3. To run our models:

```
sh run-ous.sh # run generative replay based models
```



## Note

In the generative replay module, for the generative model, we adopt the commonly used backbone generative model (WGAN) in [Deep Generative Replay](https://github.com/kuc2477/pytorch-deep-generative-replay) (DGR). As specified in the paper, the backbone AC-GAN in our FedCIL is a tiny model with the similar structure of the WGAN in the above DGR implementation.

The difference is that we add an auxiliary classification head to the top of the discriminator as introduced in [AC-GAN](https://arxiv.org/pdf/1610.09585.pdf).
