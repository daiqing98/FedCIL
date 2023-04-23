import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.model_utils import get_dataset_name
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from torch import optim

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False, my_model_name = None, unique_labels=None):
        
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.id = id  # integer
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True, shuffle = True)
        self.testloader =  DataLoader(self.test_data, self.batch_size, drop_last=True)
        
        self.testloaderfull = DataLoader(self.test_data, self.test_samples)
        self.trainloaderfull = DataLoader(self.train_data, self.train_samples, shuffle = True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        
        self.test_data_so_far = []
        self.test_data_so_far += self.test_data
        self.test_data_so_far_loader = DataLoader(self.test_data_so_far, len(self.test_data_so_far))
        
        self.test_data_per_task = []
        self.test_data_per_task.append(self.test_data)
        
        dataset_name = get_dataset_name(self.dataset)
        self.unique_labels = unique_labels
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None
        
        # continual federated learning
        self.classes_so_far = [] # all labels of a client so far 
        self.available_labels_current = [] # labels from all clients on T (current)
        self.current_labels = [] # current labels for itself
        self.classes_past_task = [] # classes_so_far (current labels excluded) 
        self.available_labels_past = [] # labels from all clients on T-1
        self.current_task = 0
        self.init_loss_fn()
        self.label_counts = {}
        self.available_labels = [] # l from all c from 0-T
        self.label_set = [i for i in range(10)]
        self.my_model_name = my_model_name
        self.last_copy = None
        self.if_last_copy = False
        self.args = args
        
    def next_task(self, train, test, label_info = None, if_label = True):
        
        if "CIFAR10" in self.args.dataset:
            optimizerD = optim.Adam(self.generator.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
            optimizerG = optim.Adam(self.generator.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

            self.generator.set_generator_optimizer(optimizerG)
            self.generator.set_critic_optimizer(optimizerD) 
            print('optimizers updated!')
        
        # update last model:
        self.last_copy  = copy.deepcopy(self.generator).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.if_last_copy = True
        
        # update generator:
        if self.my_model_name == 'fedcl':
            self.last_generator = copy.deepcopy(self.generator)
        
        # update dataset: 
        self.train_data = train
        self.test_data = test
        
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)
 
        self.trainloader = DataLoader(self.train_data, self.batch_size, drop_last=True,  shuffle = True)
        self.testloader =  DataLoader(self.test_data, self.batch_size, drop_last=True)
        
        self.testloaderfull = DataLoader(self.test_data, len(self.test_data))
        self.trainloaderfull = DataLoader(self.train_data, len(self.train_data),  shuffle = True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        
        # update classes_past_task
        self.classes_past_task = copy.deepcopy(self.classes_so_far)
        
        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            
            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])

        # update test data for CL: (classes so far)
        self.test_data_so_far += self.test_data
        self.test_data_so_far_loader = DataLoader(self.test_data_so_far, len(self.test_data_so_far))
        
        # update test data for CL: (test per task)        
        self.test_data_per_task.append(self.test_data)
        
        # update class recorder:
        self.current_task += 1
        
        return
    
    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
        
    # kd_loss
    def kd_loss(self, teacher, gen_output, generative_alpha, selected, T = 2):
                    
        # user output logp
        student_output = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
        student_output = student_output[:, selected]
        student_output_logp = F.log_softmax(student_output / T , dim=1)
   
        # global output p
        teacher_output = teacher(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
        teacher_output = teacher_output[:, selected]
        teacher_outputp = F.softmax(teacher_output / T , dim=1).clone().detach()

        # kd loss
        kd_loss = self.ensemble_loss(student_output_logp, teacher_outputp)
        kd_loss =  generative_alpha * kd_loss # debug
   
        return kd_loss
        
    def set_parameters(self, model,beta=1):
        '''
        self.model: old user model
        model: the global model on the server (new model)
        '''
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()
    
    def set_parameters_(self, model, only_critic, beta=1, mode=None, gr=False, classifier=None):
        '''
        At the beginning of this round: 
        self.model: old user model, note trained yet
        model: the global model on the server (new model)
        '''
        if gr == True:
            for old_param, new_param in zip(self.classifier.critic.parameters(), classifier.critic.parameters()):
                if beta == 1:
                    old_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()            
        
        else:
            if only_critic == True:
                for old_param, new_param in zip(self.generator.critic.parameters(), model.critic.parameters()):
                    if beta == 1:
                        old_param.data = new_param.data.clone()
                    else:
                        old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()

            else:
                 for old_param, new_param in zip(self.generator.parameters(), model.parameters()):
                    if beta == 1:
                        old_param.data = new_param.data.clone()
                    else:
                        old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()           
    
    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self, personal = True):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        test_acc = 0
        loss = 0
        
        if personal == True:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
         
        else:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
          
        return test_acc, loss, y.shape[0]

    def test_a_dataset(self, dataloader):
        '''
        test_acc: total correct samples
        loss: total loss (on a dataset) 
        y_shape: total tested samples
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item() # counts: how many correct samples
        return test_acc, loss, y.shape[0]
    
    def test_per_task(self):

        self.model.eval()
        test_acc = []
        loss = []
        y_shape = []
        
        # evaluate per task: 
        for test_data in self.test_data_per_task:
            test_data_loader = DataLoader(test_data, len(test_data))
            test_acc_, loss_, y_shape_ = self.test_a_dataset(test_data_loader)
            
            test_acc.append(test_acc_)
            loss.append(loss_)
            y_shape.append(y_shape_)
        
        return test_acc, loss, y_shape
        
    def test_all(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        test_acc = 0
        loss = 0
        for x, y in self.test_data_so_far_loader:
            x = x.to(device)
            y = y.to(device)
            
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, loss, y.shape[0]


    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], loss


    def get_next_train_batch(self, count_labels=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if count_labels:
            unique_y, counts=torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
    
    def test_(self, personal = False):
        
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic
            
        if self.my_model_name == 'fedlwf':
            model = self.model
        
        model.cuda()
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        test_acc = 0
        loss = 0
        
        if personal == True:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)
                
                if self.my_model_name == 'fedcl':
                    _, p, _ = model.forward(x)
                else:
                    p = model.forward(x)['output']      
                    
                loss += self.loss(torch.log(p), y)
                
                # mask irrevelent output logits:
                # p: 1000 * 10
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:,mask] = -9999
                
                test_acc += (torch.sum(torch.argmax(p, dim=1) == y)).item()
         
        else:
            for x, y in self.testloaderfull:
                x = x.to(device)
                y = y.to(device)

                if self.my_model_name == 'fedcl':
                    _, p, _ = model.forward(x)
                else:
                    p = model.forward(x)['output']  
                    
                loss += self.loss(torch.log(p), y)
                test_acc += (torch.sum(torch.argmax(p, dim=1) == y)).item()
          
        return test_acc, loss, y.shape[0]

    def test_a_dataset_(self, dataloader, personal = False):
        '''
        test_acc: total correct samples
        loss: total loss (on a dataset) 
        y_shape: total tested samples
        '''
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic
                
        if self.my_model_name == 'fedlwf':
            model = self.model
            
        model.cuda()
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        test_acc = 0
        loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            
            if self.my_model_name == 'fedcl':
                _, p, _ = model.forward(x)
            else:
                p = model.forward(x)['output']  
                  
            loss += self.loss(torch.log(p), y)
            
            if personal == True:
                # mask irrevelent output logits:
                # p: 1000 * 10
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:,mask] = -9999
            
            else:
                pass
            
            test_acc += (torch.sum(torch.argmax(p, dim=1) == y)).item() # counts: how many correct samples
        return test_acc, loss, y.shape[0]
    
    def test_per_task_(self, personal = False):
       
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic
            
        if self.my_model_name == 'fedlwf':
            model = self.model
            
        model.cuda()
            
        self.generator.eval()
        test_acc = []
        loss = []
        y_shape = []
        
        # evaluate per task: 
        for test_data in self.test_data_per_task:
            test_data_loader = DataLoader(test_data, len(test_data))
            test_acc_, loss_, y_shape_ = self.test_a_dataset_(test_data_loader) if personal == False else self.test_a_dataset_(test_data_loader, personal = True)
            
            test_acc.append(test_acc_)
            loss.append(loss_)
            y_shape.append(y_shape_)
        
        return test_acc, loss, y_shape
        
    def test_all_(self, personal=False, matrix=False):
        
        if self.my_model_name == 'fedcl':
            model = self.generator.critic
            if self.args.algorithm == 'FedGR':
                model = self.classifier.critic
        if self.my_model_name == 'fedlwf':
            model = self.model
            
        model.cuda()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        test_acc = 0
        loss = 0
        predicts = []
        labels = []
        
        for x, y in self.test_data_so_far_loader:
            x = x.to(device)
            y = y.to(device)
                        
            if self.my_model_name == 'fedcl':
                _, p, _ = model.forward(x)
            else:
                p = model.forward(x)['output']    

            loss += self.loss(torch.log(p), y) # if the probability of 'Y' is too small, log(p) -> INF
        
#             # debug:
#             if (self.loss(torch.log(p), y).item())**2 == float('-inf')**2: 
#                 np.set_printoptions(threshold=9999)
#                 lo = nn.NLLLoss(reduction = 'none')
                
#                 t = 0
#                 for num, sample in enumerate(lo( torch.log(p), y )):
#                     if (sample.item())**2 == float('-inf')**2: 
#                         t = num
                
#                 print(p[t])      
#                 print(y[t])
#                 exit()
                
            if personal == True:
                mask = list(set(self.label_set).difference(set(self.classes_so_far)))
                p[:,mask] = -9999
            
            else:
                pass
            
            test_acc += (torch.sum(torch.argmax(p, dim=1) == y)).item()
            
            if matrix == True:
                # confusion matrix
                predicts += torch.argmax(p, dim=1)
                labels += y
               
        
        if matrix == True:
            
            return test_acc, loss, y.shape[0], predicts, labels
        else:
            return test_acc, loss, y.shape[0]