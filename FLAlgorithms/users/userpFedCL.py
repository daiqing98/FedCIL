import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
import copy
from torch.utils.tensorboard import SummaryWriter

# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

class UserpFedCL(User):
    def __init__(self,
                 args, 
                 id, 
                 model, 
                 generator,
                 train_data, 
                 test_data,
                 label_info,
                 g,
                 use_adam=False,
                 my_model_name = None,
                 unique_labels=None,
                ):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam, my_model_name = my_model_name, unique_labels=unique_labels)
        
        self.gen_batch_size = args.gen_batch_size
        self.label_info=label_info
        self.args = args
        self.fine_tune_iters = 200
        
        # ============ AC-GAN part, class: scholar in AC-GAN ============
        self.generator = g
        self.generator_server = generator
        self.last_generator = None
        
        # init. GAN model: (deprecated)
        #self.initialize_AC_GAN(args)
    
    # ==================================== AC - GAN as clients ================================
    
    def train(
        self,
        glob_iter_,
        generator_server,
        glob_iter,
        personalized,
        verbose,
        regularization,
        importance_of_new_task = .5,
        batch_size = 32,
        iterations = 3000,
        current_task = None,
        Fedavg = False
    ):
        '''
        @ glob_iter: the overall iterations across all tasks
        
        '''
        
        # init loss:
        c_loss_all = 0
        g_loss_all = 0
        
        # preparation:
        self.clean_up_counts()
        self.model.train()
        self.generator.train()
        generator_server.eval()
        
        # variables
        current_task = self.current_task # in AC-GAN, it starts with 1
        
        # device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # mode
        self.generator.train()
        
        # =============================================== FedCL ===============================================
        if self.args.fedcl == 1:
            for iteration in range(self.local_epochs):

                samples = self.get_next_train_batch(count_labels = True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                # obtain replay samples:
                if self.last_generator is not None:
                    x_, y_ = self.last_generator.sample(self.gen_batch_size, self.classes_past_task) # last_generator cannot generate current ones
                    x_ = x_.to(device)
                    y_ = y_.to(device)
                else:
                    x_ = y_ = None
                
                x_g = y_g = None
                
                # train the model with a batch
                if self.args.offline == 0 and self.args.naive == 0: # only full when online and no naive
                    result = self.generator.train_a_batch_all(
                        available_labels = self.available_labels,
                        generator_server = generator_server,
                        glob_iter_ = glob_iter_,
                        x  = x, y = y, x_=x_, y_=y_, x_g = x_g, y_g = y_g,
                        importance_of_new_task=importance_of_new_task, classes_so_far = self.classes_so_far)
                
                else:
                    result = self.generator.train_a_batch(
                        x  = x, y = y, x_=x_, y_=y_,
                        importance_of_new_task=importance_of_new_task, classes_so_far = self.classes_so_far)
                
                # timestep for record
                time_step = iteration + glob_iter * self.local_epochs
                
            #  =============== fine-tune the classifier at the end of a TASK (CIFAR-10) ==================


        # =============================================== Fedavg ===============================================
        elif self.args.fedavg==1: 
            
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels = True)
                x, y = samples['X'].to(device), samples['y'].to(device)

                result = self.generator.train_a_batch_critic_only(
                    x, y, x_=None, y_=None,
                    importance_of_new_task=importance_of_new_task, classes_so_far = self.classes_so_far)
                
                #c_loss_all += result['c_loss']
        
        # =============================================== FedLwF ===============================================
        elif self.args.fedlwf==1:
            for iteration in range(self.local_epochs):
                samples = self.get_next_train_batch(count_labels = True)
                x, y = samples['X'].to(device), samples['y'].to(device)
                
                result = self.generator.train_a_batch_lwf(
                                        current_task = self.current_task,
                                        server_generator = generator_server,
                                        last_copy = self.last_copy,
                                        if_last_copy = self.if_last_copy,
                                        x = x, y = y, x_=None, y_=None,
                                        importance_of_new_task=importance_of_new_task, classes_so_far = self.classes_so_far)
                
                c_loss_all += result['loss_all']                
                
            c_loss_avg = c_loss_all / self.local_epochs
#             print('c_loss_avg: ', c_loss_avg)
            
    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):    
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:0 for label in range(self.unique_labels)}

# tools
# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid