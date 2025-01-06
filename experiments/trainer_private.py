import time
import os
import copy
from unittest import result
import torch
from torch import tensor
from torch.nn import parameter
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
import numpy as np

# from opacus import PrivacyEngine
# from models.losses.sign_loss import SignLoss
from models.alexnet import AlexNet
from experiments.utils import  chunks, vec_mul_ten, insta_criterion
from experiments.defense_instahide import  defense_insta

import time
import random


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TesterPrivate(object):
    def __init__(self, model, device, verbose=True):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test_signature(self, kwargs, ind):
        self.model.eval()
        avg_private = 0
        count_private = 0
        
        with torch.no_grad():
            if kwargs != None:
                if isinstance(self.model, AlexNet):
                    for m in kwargs:
                        if kwargs[m]['flag'] == True:
                            b = kwargs[m]['b']
                            M = kwargs[m]['M']

                            M = M.to(self.device)
                            if ind == 0 or ind == 1:
                                signbit = self.model.features[int(m)].scale.view([1, -1]).mm(M).sign().to(self.device)
                                #signbit = self.model.features[int(m)].scale.view([1, -1]).sign().mm(M).sign().to(self.device)
                            if ind == 2 or ind == 3:
                                w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                                signbit = w.view([1,-1]).mm(M).sign().to(self.device)
                            #print(signbit)

                            privatebit = b
                            privatebit = privatebit.sign().to(self.device)
                    
                            # print(privatebit)
        
                            detection = (signbit == privatebit).float().mean().item()
                            avg_private += detection
                            count_private += 1

                else:
                    for sublayer in kwargs["layer4"]:
                        for module in kwargs["layer4"][sublayer]:
                            if kwargs["layer4"][sublayer][module]['flag'] == True:
                                b = kwargs["layer4"][sublayer][module]['b']
                                M = kwargs["layer4"][sublayer][module]['M']
                                M = M.to(self.device)
                                privatebit = b
                                privatebit = privatebit.sign().to(self.device)

                                if module =='convbnrelu_1':
                                    scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight, dim = 0)
                                if module =='convbn_2':
                                    scale = self.model.layer4[int(sublayer)].convbn_2.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim = 0)
                               
                                if ind == 0 or ind == 1:
                                    signbit = scale.view([1, -1]).mm(M).sign().to(self.device)
                                    #signbit = scale.view([1, -1]).sign().mm(M).sign().to(self.device)

                                if ind == 2 or ind == 3:
                                    signbit = conv_w.view([1,-1]).mm(M).sign().to(self.device)
                            #print(signbit)
                            # print(privatebit)
                                detection = (signbit == privatebit).float().mean().item()
                                avg_private += detection
                                count_private += 1

        if kwargs == None:
            avg_private = None
        if count_private != 0:
            avg_private /= count_private

        return avg_private

class TrainerPrivate(object):
    def __init__(self, model, train_set, device, dp, sigma,num_classes,defense=None,klam=3,up_bound=0.65,mix_alpha=0.01):
        self.model = model
        self.device = device
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma
        self.defense=defense
        self.klam=klam
        self.up_bound=up_bound
        self.mix_alpha=mix_alpha
        
        self.num_classes=num_classes
        self.train_loader=train_set
        self.batch_size=100
    

    def mixup_data(self, x, y,alpha):
        
        use_cuda=True
        # print('alpha:',alpha)

        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        # print('lam:',lam)
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        # print('index:',index)
        return mixed_x, y_a, y_b, lam



    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def _local_update_noback(self, dataloader, local_ep, lr, optim_choice, sampling_proportion):
        
        if optim_choice=="sgd":
        
            self.optimizer = optim.SGD(self.model.parameters(),
                                lr,
                                momentum=0.9,
                                weight_decay=0.0005)
        else:
             self.optimizer = optim.AdamW(self.model.parameters(),
                                lr,
                                weight_decay=0.0005)
                                  
        epoch_loss = []
        cos_scores=[]
        train_ldr = dataloader 

        for epoch in range(local_ep):
            
            loss_meter = 0
            acc_meter = 0
            sample_grads=[] 
            total=0
            correct=0
            if self.defense!='instahide':
                iteration=0
                for batch_idx, (x, y) in enumerate(train_ldr):
                    sample_batch_grads=[]
                    #print("batch_idx:{}\n x:{} \n y:{}\n".format(batch_idx,x,y))
                    x, y = x.to(self.device), y.to(self.device)
                    if self.defense=='mix_up':
                        # print("mix_up training...")
                        inputs, targets_a, targets_b, lam = self.mixup_data(x, y,self.mix_alpha)
                                                            
                        # inputs, targets_a, targets_b = map(Variable, (inputs,
                        #                                     targets_a, targets_b))
                        self.optimizer.zero_grad()
                        # loss = torch.tensor(0.).to(self.device)

                        pred = self.model(x)
                        loss = self.mixup_criterion( F.cross_entropy, pred, targets_a, targets_b, lam)
                        _, predicted = torch.max(pred.data, 1)
                        total += y.size(0)
                        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
                        #loss += F.cross_entropy(pred, y)

                        acc_meter+=100* correct/total
                        loss.backward()

                        self.optimizer.step() 
                        loss_meter += loss.item()
                    elif self.defense=='instahide':

                        inputs,targets,lams=defense_insta.generate_sample(dataloader,self.klam,)
                        #inputs,targets,lams=self.instahide_data(x,y,self.klam)
                        self.optimizer.zero_grad()
                        inputs=Variable(inputs)
                        outputs=self.model(inputs)

                        loss= insta_criterion(outputs, targets,lams,self.klam,self.num_classes)
                        loss_meter+= loss.data.item()
                        loss.backward()
                        self.optimizer.step() 


                    else:
                        self.optimizer.zero_grad()

                        loss = torch.tensor(0.).to(self.device)

                        pred = self.model(x)
                        loss += F.cross_entropy(pred, y)
                        acc_meter += accuracy(pred, y)[0].item()
                        loss.backward()

                        self.optimizer.step() 
                        loss_meter += loss.item()

                    # sampling num = batch_size * sampling_iteration = 100 * 25 = 2500  
                    iteration+=1
                    # print("iteration:",iteration)
                    if  iteration == int(sampling_proportion * len(train_ldr)):
                        break

                loss_meter /= len(train_ldr)
                acc_meter /= len(dataloader)
                epoch_loss.append(loss_meter)
            else:
                train_loss, correct, total = 0, 0, 0
                self.optimizer.zero_grad()
                instahide=defense_insta(dataloader,self.klam,self.up_bound)

                self.model.train()
        
                mix_inputs_all, mix_targets_all, lams = instahide.generate_sample(dataloader, self.klam)
                # print(len(mix_inputs_all))
                seq = random.sample(range(len(mix_inputs_all)), len(mix_inputs_all))
                bl = list(chunks(seq, self.batch_size))
                #print('bl:',len(bl))

                for batch_idx in range(len(bl)):
                    b = bl[batch_idx]
                    #print('b:',b)
                    inputs = torch.stack([mix_inputs_all[i] for i in b])
                    lam_batch = torch.stack([lams[i] for i in b])

                    mix_targets = []
                    #print( [mix_targets_all[ik][ib].long().item() for ib in b])
                    for ik in range(self.klam):
                        #print( [mix_targets_all[ik][ib].long().to(self.device) for ib in b])
                        mix_targets.append(
                            torch.stack(
                                [mix_targets_all[ik][ib].long().to(self.device) for ib in b]).to(self.device))
                                #[mix_targets_all[ik][ib].long().to(device) for ib in b]))
                    targets_var = [Variable(mix_targets[ik]) for ik in range(self.klam)]

                    inputs = Variable(inputs)
                    outputs = self.model(inputs)
                    loss = insta_criterion(outputs, targets_var, lam_batch,self.klam, self.num_classes)
                    train_loss += loss.data.item()
                    total += self.batch_size
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss /= len(bl)
                epoch_loss.append(train_loss)
                        
        if self.dp:
            # print('DP setting ......')
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)
        
        
        return self.model.state_dict(), np.mean(epoch_loss)
    
    def test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)
        
                pred = self.model(data)  # test = 4
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter

    def fake_test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)
        
                pred = self.model(data)  # test = 4
                #loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred_result = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                fake_result = pred_result.view_as(target)
               
                loss_meter += F.cross_entropy(pred, fake_result, reduction='sum').item()
                acc_meter += pred_result.eq(target.view_as(pred_result)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter

 
