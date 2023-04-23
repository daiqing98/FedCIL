from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import os
import os.path
from FLAlgorithms.generator import gan
import numpy as np
from torch.nn import functional as F
import torchvision

EPSILON = 1e-16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, master_rank):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.master_rank = master_rank
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels==c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label):

        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()

class WGAN(nn.Module):
    def __init__(self, z_size,
                 image_size, 
                 image_channel_size,
                 c_channel_size, 
                 g_channel_size,
                 dataset,
                ):

        super().__init__()
    
        # loss functions
        self.dis_criterion = nn.BCELoss()
        self.aux_criterion = nn.NLLLoss()
        self.z_size = z_size
        self.image_size = image_size 
        self.image_channel_size = image_channel_size
        self.c_channel_size = c_channel_size
        self.g_channel_size = g_channel_size
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        
        # Build backbone
        if dataset == 'MNIST':
            self.critic = gan.Critic(
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.c_channel_size,
            )
            self.generator = gan.Generator(
                z_size=self.z_size,
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.g_channel_size,
            )
            self.num_classes = 10
            
        if dataset == 'CIFAR10':
            self.critic = gan.Critic_CIFAR(
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.c_channel_size,
          )
            self.generator = gan.Generator_CIFAR(
                z_size=self.z_size,
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.g_channel_size,
          )
            self.num_classes = 10
                
        if dataset == 'EMNIST-L':
            self.critic = gan.Critic(
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.c_channel_size,
                num_classes = 26
            )
            self.generator = gan.Generator(
                z_size=self.z_size,
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.g_channel_size,
            )
            self.num_classes = 26
        
        if dataset == 'EMNIST-B':
            self.critic = gan.Critic(
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.c_channel_size,
                num_classes = 47
            )
            self.generator = gan.Generator(
                z_size=self.z_size,
                image_size=self.image_size,
                image_channel_size=self.image_channel_size,
                channel_size=self.g_channel_size,
            )
            self.num_classes = 47
        
        # training related components that should be set before training.
        self.generator_optimizer = None
        self.critic_optimizer = None
        self.lamda = None
        
        # Build D2DCE for ReACGAN
        self.D2DCE = Data2DataCrossEntropyLoss(self.num_classes, 0.5, 0.98, torch.device('cuda:0')) # default parameter in that paper
        
    def kd_loss(self, logits_a, logits_b):
        return F.mse_loss(logits_a, logits_b)
    
    def train_a_batch(self, x, y,
                      classes_so_far,
                      x_=None, y_=None, 
                      importance_of_new_task=.5, 
                      x_g =None, y_g = None,
                     ):
        
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()
        c_logits_replay = None

        # =============== update D ==============
        # run the critic on the real data.
        c_loss_real, g_real, c_logits, aux = self._c_loss(x, y, classes_so_far, return_g=True, return_aux=True)
        
        # ==============
        # 1. AC-GAN loss
        # ==============
        
        # run the critic on the replayed data.
        if x_ is not None and y_ is not None:
            c_loss_replay, g_replay, c_logits_replay, aux = self._c_loss(x_, y_, classes_so_far, return_g=True, return_aux=True)
 
            c_loss = (
                importance_of_new_task * c_loss_real +
                (1-importance_of_new_task) * c_loss_replay
            )
    
        else:
            c_loss = c_loss_real
        
        c_loss = c_loss
        # updation
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()
        
        # =============== update G ==============
        self.generator_optimizer.zero_grad()
        
        # 1. AC-GAN loss:
        g_loss, g_logits = self._g_loss(x, y, classes_so_far)
        g_loss.backward()
        self.generator_optimizer.step()   
        return {'c_loss': c_loss.item(), 'g_loss': g_loss.item(), 'aux_f': aux[0].item(), 'aux_r': aux[1].item(), 'features': aux[2],}   

    
    

    def sample(self, size, classes_so_far):
        noise, aux_label, _ = self.generate_noise_with_classes(size, classes_so_far)
        
        fake = self.generator(noise.to(torch.device('cuda:0')))
        
        return fake, aux_label
    
    def set_generator_optimizer(self, optimizer):
        self.generator_optimizer = optimizer

    def set_critic_optimizer(self, optimizer):
        self.critic_optimizer = optimizer

    def set_critic_updates_per_generator_update(self, k):
        self.critic_updates_per_generator_update = k

    def set_lambda(self, l):
        self.lamda = l

    def _noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.to(torch.device('cuda:0'))

    
############################################################


    def _c_loss(self, x, y, classes_so_far, return_g=False, return_aux=False, return_feature=False):  
        
        # info
        batch_size = x.size(0)
        
        # generate label:
        dis_label = torch.FloatTensor(batch_size)
        dis_label = dis_label.to(torch.device('cuda:0'))
        y = y.to(torch.device('cuda:0'))
        dis_label = Variable(dis_label)
        dis_label.data.fill_(1)
        
        # train with real data:
        dis_output, aux_output, logits_real, feature = self.critic(x, if_features=True)
        
        # data transform
        y_hot = torch.nn.functional.one_hot(y.to(torch.int64), self.num_classes).float()
        embed = self.critic.linear2(feature)
        embed = F.normalize(embed, dim=1)
        proxy = self.critic.embedding(y_hot)
        proxy = F.normalize(proxy, dim=1)
        label = y 
        
        # loss 
        dis_errD_real = self.D2DCE(embed, proxy, label)
        aux_errD_real = self.aux_criterion(torch.log(aux_output), y)
        loss_c_real = dis_errD_real + aux_errD_real 
        
        # train with generated data:
        # generate noise:
        noise, aux_label, dis_label = self.generate_noise_with_classes(batch_size, classes_so_far)
        fake = self.generator(noise.to(torch.device('cuda:0')))
        dis_output, aux_output, logits, feature_g= self.critic(fake.detach(), if_features=True) # G will not be updated
        
        # data transform
        aux_label_hot = torch.nn.functional.one_hot(aux_label.to(torch.int64), self.num_classes).float()
        embed_g = self.critic.linear2(feature_g)
        embed_g = F.normalize(embed_g, dim=1)
        proxy_g = self.critic.embedding(aux_label_hot)
        proxy_g = F.normalize(proxy_g, dim=1)
        label = aux_label
        
        # loss functinos
        dis_errD_fake = self.D2DCE(embed, proxy, label)
        aux_errD_fake = self.aux_criterion(torch.log(aux_output), y)
        loss_c_fake = dis_errD_fake + aux_errD_fake 
        
        loss_c_fake = dis_errD_fake + aux_errD_fake
        loss_c_all = loss_c_fake + loss_c_real
        
        if return_g:
            if return_aux == True:
                return loss_c_all, fake, logits_real, (aux_errD_fake, aux_errD_real, feature)
            else:
                return loss_c_all, fake, logits_real
        else:
            if return_aux == True:
                return loss_c_all, logits_real, (aux_errD_fake, aux_errD_real, feature)
            else:
                return loss_c_all, logits_real
        
    def generate_noise_with_classes(self, batch_size, classes_so_far, label = None):
        
        noise = torch.FloatTensor(batch_size, self.z_size, 1, 1)
        dis_label = torch.FloatTensor(batch_size)
        aux_label = torch.LongTensor(batch_size)
        real_label = 1
        fake_label = 0
        
        dis_label, aux_label = dis_label.to(torch.device('cuda:0')), aux_label.to(torch.device('cuda:0'))
        
        # train with real data:
        # define variables
        noise = Variable(noise)
        dis_label = Variable(dis_label)
        aux_label = Variable(aux_label)

        # to obtain the noise:
        # why 1,1?????
        noise.data.resize_(batch_size, self.z_size, 1, 1).normal_(0, 1)
        
        label = np.random.choice(classes_so_far, batch_size) if label is None else label
        
        noise_ = np.random.normal(0, 1, (batch_size, self.z_size))
        
        class_onehot = np.zeros((batch_size, self.num_classes))
        
        class_onehot[np.arange(batch_size), label] = 1
        
        noise_[np.arange(batch_size), :self.num_classes] = class_onehot[np.arange(batch_size)]
        
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, self.z_size, 1, 1))
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        
        # generate images with noise & labels
        noise = torch.squeeze(noise)
        dis_label.data.fill_(fake_label)
        
        return noise, aux_label, dis_label

    def _g_loss(self, x, y, classes_so_far, return_g=False):
        
        # info
        batch_size = x.size(0)
        
        # generate noise, aux_label is labeled FALSE
        noise, aux_label, _ = self.generate_noise_with_classes(batch_size = batch_size, classes_so_far = classes_so_far)
        
        # prepare new dis_label:
        dis_label = torch.FloatTensor(batch_size)
        dis_label = dis_label.to(torch.device('cuda:0'))
        dis_label = Variable(dis_label)
        dis_label.data.fill_(1)
        
        fake = self.generator(noise.to(torch.device('cuda:0')))        
 
        dis_output, aux_output, logits = self.critic(fake)
        
        #print('g_loss_dis_output: ' + str(dis_output))
        #print('g_loss_dis_label: ' + str(dis_label))
        #print('g_loss_noise: ' + str(noise))
        #print('g_loss_aux_label: ' + str(aux_label))
        
        # NLLL loss
        # aux_output = torch.log(aux_output)
        
        dis_errG = self.dis_criterion(dis_output, dis_label)    
        aux_errG = self.aux_criterion(torch.log(aux_output), aux_label)
        
        loss_g = dis_errG + aux_errG
        
        #print('g_loss_aux_output: ' + str(aux_output))
        #print('g_loss_aux_errG: ' + str(aux_errG))
        
        if return_g:
            return (loss_g, g) 
        else:
            return loss_g, logits
        
    def visualize(self, sample_size = 16, path = './images'):
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data, label = self.sample(sample_size, 1 )
        
        torchvision.utils.save_image(
        data,
        path + '.jpg',
        nrow=6,
    )
        
        print('image is saved!')
                
    def train_a_batch_critic_only(self,
                                  x, y,
                                  classes_so_far,
                                  x_=None, y_=None, 
                                  importance_of_new_task=.5):
    
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()
        
        # ===================
        # 1. prediction loss
        # ====================
        
        dis_output, aux_output, logits_real = self.critic(x)
        c_loss_real = self.aux_criterion(torch.log(aux_output), y)
        
        
        # run the critic on the replayed data.
        if x_ is not None and y_ is not None:
                dis_output, aux_output, logits_real = self.critic(x_)
                c_loss_replay = self.aux_criterion(torch.log(aux_output, y_))
        
                c_loss = (importance_of_new_task * c_loss_real + (1-importance_of_new_task) * c_loss_replay)
        
        else:
            c_loss = c_loss_real
        
        # updation
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_optimizer.step()        

        return {'c_loss': c_loss.item()}   

    def train_a_batch_lwf(self,
                          current_task,
                          server_generator,
                          last_copy,
                          if_last_copy,
                          x, y,
                          classes_so_far,
                          x_=None, y_=None, 
                          importance_of_new_task=.5):
    
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()
        
        T = 2
        
        # ===================
        # 1. prediction loss
        # ===================
        
        dis_output, aux_output, logits_real = self.critic(x)
        c_loss_real = self.aux_criterion(torch.log(aux_output), y)
        
        # =======================
        # 2. kd: server -> client
        # =======================

        if current_task > 0:
            user_logit_logp = torch.log(F.softmax(logits_real/T, dim=1))
            
            dis_output, aux_output, server_logit = server_generator.critic(x)
            server_logit_p = F.softmax(server_logit/T, dim=1).clone().detach()
            
            kd_loss_server = self.ensemble_loss(user_logit_logp, server_logit_p)
        
        else:
            kd_loss_server = 0
        
        #=================================
        # 3. KD loss : last copy -> client
        #================================= 
        if if_last_copy:
            dis_output, aux_output, copy_logit = last_copy.critic(x)
            copy_logit_p = F.softmax(copy_logit/T, dim=1).clone().detach()
            
            kd_loss_copy = self.ensemble_loss(user_logit_logp, copy_logit_p)
        
        else:
            kd_loss_copy = 0
        
        if current_task > 0:
            alpha = 0.33
            beta = 0.33
        
        else:
            alpha = 1
            beta = 0
        
        loss_all = alpha * c_loss_real + beta * kd_loss_copy + (1 - alpha - beta) * kd_loss_server  
        
        # updation
        self.critic_optimizer.zero_grad()
        loss_all.backward()
        self.critic_optimizer.step()        

        return {'loss_all': loss_all.item()}   

    # User local training
    def train_a_batch_all(self, x, y,
                      available_labels,
                      classes_so_far,
                      generator_server,
                      glob_iter_,
                      x_=None, y_=None, 
                      importance_of_new_task=.5, 
                      x_g =None, y_g = None,
                     ):
        
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()
        c_logits_replay = None

        # =============== update D ==============
        # run the critic on the real data.
        c_loss_real, g_real, c_logits, aux = self._c_loss(x, y, classes_so_far, return_g=True, return_aux=True)
        
        # ==============
        # 1. AC-GAN loss
        # ==============
        
        # run the critic on the replayed data.
        if x_ is not None and y_ is not None:
            c_loss_replay, g_replay, c_logits_replay, aux = self._c_loss(x_, y_, classes_so_far, return_g=True, return_aux=True)
 
            c_loss = (
                importance_of_new_task * c_loss_real +
                (1-importance_of_new_task) * c_loss_replay
            )
    
        else:
            c_loss = c_loss_real
        
        # ============================
        # 2. kd loss: D
        # ============================
        batch_size = y.size(0)
        if glob_iter_ != 0:

            noise, aux_label, _ = self.generate_noise_with_classes(batch_size, classes_so_far = None, label = y.cpu().detach().numpy())
            fake_server = generator_server.generator(noise.to(torch.device('cuda:0')))

            # client output with fake_server:
            _, p, logits = self.critic(fake_server.detach())
            # client output with real:
            _, p_real, logits_real = self.critic(x)
            # kd loss:
            kd_loss_d = self.ensemble_loss(torch.log(p_real), p) 
        else:
            kd_loss_d = 0
        
        # ============================
        # 3. kd loss: G
        # ============================
        if glob_iter_ != 0:
            # fake_server
            noise, aux_label, _ = self.generate_noise_with_classes(batch_size, classes_so_far = available_labels)
            fake_server = generator_server.generator(noise.to(torch.device('cuda:0')))        
            
            # fake_own:
            noise, aux_label, _ = self.generate_noise_with_classes(batch_size, classes_so_far = None, label = aux_label.cpu().detach().numpy())
            fake_own = self.generator(noise.to(torch.device('cuda:0')))   
            
            # client output with fake_server:
            _, p_server, logits_server = self.critic(fake_server.detach())
            # client output with its own:
            _, p, logits = self.critic(fake_own.detach())
            # kd loss:
            kd_loss_g_1 = self.ensemble_loss(torch.log(p), p_server)
            # classification loss:
            kd_loss_g_2 = self.aux_criterion(torch.log(p_server),aux_label)
            
            kd_loss_g = kd_loss_g_1 + kd_loss_g_2
        else:
            kd_loss_g = kd_loss_g_1 = kd_loss_g_2 = 0 
        
        c_loss = c_loss + kd_loss_d * 0 + kd_loss_g_1* 0 + kd_loss_g_2*0 # ablation
        #c_loss = c_loss
        
        # updation
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()
        
        # =============== update G ==============
        self.generator_optimizer.zero_grad()
        
        # 1. AC-GAN loss:
        g_loss, g_logits = self._g_loss(x, y, classes_so_far)
        
        g_loss.backward()
        self.generator_optimizer.step()
        

        
        return {'features': aux[2],'aux_f': aux[0].item(), 'aux_r': aux[1].item(), 'c_loss': c_loss.item(), 'g_loss': g_loss.item(), 'kd_loss_d': kd_loss_d.item() if glob_iter_ != 0 else 0, 'kd_loss_g': kd_loss_g.item() if glob_iter_ != 0 else 0}   