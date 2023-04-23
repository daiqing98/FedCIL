import torch
import os
import numpy as np
import h5py
from utils.model_utils import get_dataset_name, RUNCONFIGS
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from utils.model_utils import get_log_path, METRICS
from torch import optim

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

class Server:
    def __init__(self, args, model, seed):

        # Set up the main attributes
        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.K = args.K
        self.args=args
        
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        
        self.users = []
        self.selected_users = []
        self.num_users = args.num_users
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.personalized = 'pFed' in self.algorithm
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key:[] for key in METRICS}
        self.timestamp = None
        self.save_path = args.result_path
        os.system("mkdir -p {}".format(self.save_path))
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1, 0.02)
            m.bias.data.fill_(0)
    
    def initialize_AC_GAN_CIFAR(self, args):
        
        self.generator.critic.apply(self.weights_init)
        self.generator.generator.apply(self.weights_init)
        
        optimizerD = optim.Adam(self.generator.critic.parameters(), lr=args.lr_CIFAR, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.generator.generator.parameters(), lr=args.lr_CIFAR, betas=(0.5, 0.999))
        
        self.generator.set_generator_optimizer(optimizerG)
        self.generator.set_critic_optimizer(optimizerD) 
        
    def initialize_AC_GAN(self, args):
        # define solver criterion and generators for the scholar model.
        beta1 = args.beta1
        beta2 = args.beta2
        lr = args.lr
        weight_decay = args.weight_decay
        
        generator_g_optimizer = optim.Adam(
            self.generator.generator.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )
        generator_c_optimizer = optim.Adam(
            self.generator.critic.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )

        self.generator.set_lambda(args.generator_lambda)
        self.generator.set_generator_optimizer(generator_g_optimizer)
        self.generator.set_critic_optimizer(generator_c_optimizer) 
        
        # initialize model parameters
        self.gaussian_intiailize(self.generator, std=.02)
    
    def initialize_Classifier(self, args):
        
        beta1 = args.beta1
        beta2 = args.beta2
        lr = args.lr
        weight_decay = args.weight_decay
        
        self.classifier.optimizer = optim.Adam(
            self.classifier.critic.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        ) 
        
        # initialize model parameters
        self.gaussian_intiailize(self.classifier.critic, std=.02)
        
        return
    
    def gaussian_intiailize(self, model, std=.01):
        
        # batch norm is not initialized 
        modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
        parameters = [p for m in modules for p in m.parameters()]
        
        for p in parameters:
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0, std=0.02)
            else:
                nn.init.constant_(p, 0)
        
        # normalization for batch norm
        modules = [m for n, m in model.named_modules() if 'bn' in n]
        
        for m in modules:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def init_ensemble_configs(self):
        #### used for ensemble learning ####
        dataset_name = get_dataset_name(self.dataset)
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr) )
        print("ensemble_batch_size: {}".format(self.ensemble_batch_size) )
        print("unique_labels: {}".format(self.unique_labels) )


    def if_personalized(self):
        return 'pFed' in self.algorithm or 'PerAvg' in self.algorithm

    def if_ensemble(self):
        return 'FedE' in self.algorithm

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        
        for user in users:
            if mode == 'all': # share all parameters
                user.set_parameters(self.model,beta=beta)
            else: # share a part parameters
                user.set_shared_parameters(self.model,mode=mode)

    def send_parameters_(self, mode='all', beta=1, selected=False, only_critic=False, gr=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        
        for user in users:
            if gr == True:
                user.set_parameters_(self.generator,beta=beta, only_critic = only_critic, mode = mode, gr=gr, classifier=self.classifier) # classifier: from server
            else:
                user.set_parameters_(self.generator,beta=beta, only_critic = only_critic, mode = mode, gr=gr)

    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio

    def add_parameters_(self, user, ratio, partial=False, gr=False):

        if gr == False:
            for server_param, user_param in zip(self.generator.parameters(), user.generator.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.classifier.critic.parameters(), user.classifier.critic.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        
    def aggregate_parameters(self,partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        
        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data) # initilize w with zeros
        
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # length of the train data for weighted importance
        
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train,partial=partial) 

    def aggregate_parameters_(self,partial=False, gr=False):
        '''
        Clients -> Server model
        '''
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        
            
        if gr == False:
            for param in self.generator.parameters():
                param.data = torch.zeros_like(param.data) # initilize w with zeros
        else:
            for param in self.classifier.critic.parameters():
                param.data = torch.zeros_like(param.data) # initilize w with zeros
        
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # length of the train data for weighted importance
        
        for user in self.selected_users:
            self.add_parameters_(user, user.train_samples / total_train,partial=partial, gr=gr) 


    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))


    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users, [i for i in range(len(self.users))]

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)


    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()


    def save_results(self, args):
        alg = get_log_path(args, args.algorithm, self.seed, args.gen_batch_size)
        with h5py.File("./{}/{}.h5".format(self.save_path, alg), 'w') as hf:
            for key in self.metrics:
                hf.create_dataset(key, data=self.metrics[key])
            hf.close()
        

    def test(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    def test_all(self, selected=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test_all()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses    
    
    def test_per_task(self, selected = False):
        '''
        tests latest model on leanrt tasks
        '''
        accs = {}
        
        users = self.selected_users if selected else self.users
        for c in users:
            accs[c.id] = []
                
            ct, c_loss, ns = c.test_per_task()
            
            # per past task: 
            for task in range(len(ct)):
                acc = ct[task] / ns[task]
                accs[c.id].append(acc)
        
        return accs

    def test_(self, selected=False, personal = False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, c_loss, ns = c.test_(personal = personal)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    def test_all_(self, selected=False, personal = False, matrix=False):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        preds = []
        labels = []
        
        users = self.selected_users if selected else self.users
        for c in users:
            if matrix == False:
                ct, c_loss, ns = c.test_all_(personal = personal)
            else:
                ct, c_loss, ns, pred, label = c.test_all_(personal = personal, matrix=True)
            
                preds += pred
                labels += label
            
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(c_loss)
        ids = [c.id for c in self.users]
        
        if matrix == False:
            return ids, num_samples, tot_correct, losses
        else:
            return ids, num_samples, tot_correct, losses, preds, labels
    
    def test_per_task_(self, selected = False, personal = False):
        '''
        tests latest model on leanrt tasks
        '''
        accs = {}
        
        users = self.selected_users if selected else self.users
        for c in users:
            accs[c.id] = []
                
            ct, c_loss, ns = c.test_per_task_(personal = personal)
            
            # per past task: 
            for task in range(len(ct)):
                acc = ct[task] / ns[task]
                accs[c.id].append(acc)
        
        return accs    
    

    def test_personalized_model(self, selected=True):
        '''
        tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        users = self.selected_users if selected else self.users
        for c in users:
            ct, ns, loss = c.test_personalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(loss)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def evaluate_personalized_model(self, selected=True, save=True):
        stats = self.test_personalized_model(selected=selected)
        test_ids, test_num_samples, test_tot_correct, test_losses = stats[:4]
        glob_acc = np.sum(test_tot_correct)*1.0/np.sum(test_num_samples)
        test_loss = np.sum([x * y for (x, y) in zip(test_num_samples, test_losses)]).item() / np.sum(test_num_samples)
        if save:
            self.metrics['per_acc'].append(glob_acc)
            self.metrics['per_loss'].append(test_loss)
        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, test_loss))


    def evaluate_ensemble(self, selected=True):
        self.model.eval()
        users = self.selected_users if selected else self.users
        test_acc=0
        loss=0
        for x, y in self.testloaderfull:
            target_logit_output=0
            for user in users:
                # get user logit
                user.model.eval()
                user_result=user.model(x, logit=True)
                target_logit_output+=user_result['logit']
            target_logp=F.log_softmax(target_logit_output, dim=1)
            test_acc+= torch.sum( torch.argmax(target_logp, dim=1) == y ) #(torch.sum().item()
            loss+=self.loss(target_logp, y)
        loss = loss.detach().numpy()
        test_acc = test_acc.detach().numpy() / y.shape[0]
        self.metrics['glob_acc'].append(test_acc)
        self.metrics['glob_loss'].append(loss)
        print("Average Accuracy = {:.4f}, Loss = {:.2f}.".format(test_acc, loss))


    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
   
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
    
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
        
    def evaluate_all(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_all(selected=selected)
        
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

        
    def evaluate_per_client_per_task(self, save=True, selected=False):
        accs = self.test_per_task()
        
        for k, v in accs.items():
            print(k)
            print(v)

    def evaluate_(self, save=True, selected=False, personal = False):
        
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test_(selected=selected, personal = personal)
   
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
    
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        print("Average Global Accuracy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
    
    def write(self, accuracy, file = None , mode = 'a'):
        with open(file, mode) as f:
                line = str(accuracy) + '\n'
                f.writelines(line)
                
    
    def evaluate_all_(self, save=True, selected=False, personal=False, matrix=False):
        '''
        test_all_() returns lists of a certain info. of all Clients. [data of client_1, d_o_c_2, ...]
        '''
        
        if matrix == False:
            test_ids, test_samples, test_accs, test_losses = self.test_all_(selected=selected, personal = personal)
        else:
            test_ids, test_samples, test_accs, test_losses, preds, labels = self.test_all_(selected=selected, personal = personal, matrix=True)
            # save pdf
            save_matrix(preds, labels)
        
        glob_acc = np.sum(test_accs)*1.0/np.sum(test_samples)
        glob_loss = np.sum([x * y.detach().to(torch.device('cpu')) for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
  
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        if personal == False:
            print("Average Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
                    
        else: 
            print("Average Accuracy (classes so far) = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))
                    
            
        
    def evaluate_per_client_per_task_(self, save=True, selected=False, personal = False):
        
        accs = self.test_per_task_(personal = personal)
        
        for k, v in accs.items():
            print('Client-' + str(k)[-1] + ': '+str(v[0]))

def save_matrix(preds, labels):
    p = []
    for item in preds:
        p.append(item.cpu().numpy())

    l = []
    for item in labels:
        l.append(item.cpu().numpy())
    
    s = set()
    for item in l:
        s.add(int(item))
    
    s = list(s)
    
    sns.set()
    f,ax=plt.subplots()
    df= confusion_matrix(l, p, labels=s)
    
    min_ = 0
    max_ = 0

    for row in df:
        for v in row:
            if v >= max_:
                max_ = v
            if v <= min_:
                min_ = v

    df_n = (df - min_) / (max_ - min_)

    sns.heatmap(df_n,annot=False,ax=ax, yticklabels=True, xticklabels=True,) #画热力图
    name = 'None'
    plt.savefig('matrix/' + name)
