import sys
import os
import numpy as np
import torch
from collections import ChainMap
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import copy
import scipy
import time
import json
import math
import random

import warnings
warnings.filterwarnings('ignore')

def liratio(mu_in,mu_out,var_in,var_out,new_samples):
    #l_in=np.sqrt(var_in)*np.exp(-((new_samples-mu_in)*(new_samples-mu_in))/(2*var_in+1e-3) )
    #l_out=np.sqrt(var_out)*np.exp(-((new_samples-mu_out)*(new_samples-mu_out))/(2*var_out+1e-3))
    l_out=scipy.stats.norm.cdf(new_samples,mu_out,np.sqrt(var_out))
    return l_out

@ torch.no_grad()
def hinge_loss_fn(x,y):
    x,y=copy.deepcopy(x).cuda(),copy.deepcopy(y).cuda()
    mask=torch.eye(x.shape[1],device="cuda")[y].bool()
    tmp1=x[mask]
    x[mask]=-1e10
    tmp2=torch.max(x,dim=1)[0]
    # print(tmp1.shape,tmp2.shape)
    return (tmp1-tmp2).cpu().numpy()

def ce_loss_fn(x,y):
    loss_fn=torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fn(x,y)

def extract_hinge_loss(i):
    val_dict={}
    val_index=i["val_index"]
    val_hinge_index=hinge_loss_fn(i["val_res"]["logit"] , i["val_res"]["labels"] )
    for j,k in zip(val_index,val_hinge_index):
        if j in val_dict:
            val_dict[j].append(k)
        else:
            val_dict[j]=[k]

    train_dict={}
    train_index=i["train_index"]
    train_hinge_index=hinge_loss_fn(i["train_res"]["logit"] , i["train_res"]["labels"] )
    for j,k in zip(train_index,train_hinge_index):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j]=[k]
    
    test_dict={}
    test_index=i["test_index"]
    test_hinge_index=hinge_loss_fn(i["test_res"]["logit"] , i["test_res"]["labels"] )
    for j,k in zip(test_index,test_hinge_index):
        if j in test_dict:
            test_dict[j].append(k)
        else:
            test_dict[j]=[k]

    return (val_dict,train_dict,test_dict)

def plot_auc(name,target_val_score,target_train_score,epoch): 
    # print('target_val_score.shape:',target_val_score.shape)
    # indices = random.sample([i for i in range(0,target_val_score.shape[0])], target_train_score.shape[0])
    # target_val_score = torch.index_select(target_val_score, 0, torch.tensor(indices))
    # print('after sampling target_val_score.shape:',target_val_score.shape)


    fpr, tpr, thresholds = metrics.roc_curve(torch.cat( [torch.zeros_like(target_val_score),torch.ones_like(target_train_score)] ).cpu().numpy(), torch.cat([target_val_score,target_train_score]).cpu().numpy())
    auc=metrics.auc(fpr, tpr)
    log_tpr,log_fpr=np.log10(tpr),np.log10(fpr)
    log_tpr[log_tpr<-5]=-5
    log_fpr[log_fpr<-5]=-5
    log_fpr=(log_fpr+5)/5.0
    log_tpr=(log_tpr+5)/5.0
    log_auc=metrics.auc( log_fpr,log_tpr )

    tprs={}
    for fpr_thres in [10, 1, 0.1,0.02,0.01,0.001,0.0001]:
        tpr_index = np.sum(fpr<fpr_thres)
        tprs[str(fpr_thres)]=tpr[tpr_index-1]
    return auc,log_auc,tprs

def common_attack(f,K,epch,extract_fn=None):
    accs=[]
    target_res=torch.load(f.format(0,epch))

    # target_train_loss=hinge_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
    # target_test_loss=hinge_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
    
    target_train_loss=-ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
    if MODE=="test":
        target_test_loss=-ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
    elif MODE=="val":
        target_test_loss=-ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] )

    auc,log_auc,tprs=plot_auc("common",torch.tensor(target_test_loss),torch.tensor(target_train_loss),epch)
    print("__"*10,"common")
    print(f"tprs:{tprs}", log_auc)
    # print("test_acc:",target_res[taret_idx])
    print("__"*10,)

    return accs,tprs,auc,log_auc,(target_test_loss,target_train_loss)

def lira_attack_ldh_cosine(f,epch,K, save_dir, extract_fn=None,attack_mode="cos"):
    # attack_mode="cos"
    print('******************************************************')
    print('************','Epch:',epch,' attack_mode:',attack_mode,'**************')
    print('******************************************************')
    save_log=save_dir + '/' + f'attack_sel{select_mode}_{select_method}_{attack_mode}.log'
    accs=[]
    training_res=[]
    for i in range(K):
        # print(i,epch)
        # training_res.append(torch.load(f.format(i,epch),map_location=lambda storage, loc: storage))
        training_res.append(torch.load(f.format(i,epch)))
        accs.append(training_res[-1]["test_acc"])
    
    target_idx=0
    val_idx = 1
    target_res=training_res[target_idx]
    shadow_res=training_res[val_idx:]
    #print(target_res["tarin_cos"])
    if attack_mode=="cos":
        target_train_loss=torch.tensor(target_res["tarin_cos"]).cpu().numpy()
        if MODE=="test":
            target_test_loss=torch.tensor(target_res["test_cos"]).cpu().numpy()
        elif MODE=="val":
            target_test_loss=torch.tensor(target_res["val_cos"]).cpu().numpy()
        elif MODE =='mix':
            random_indices = torch.randperm(target_res["test_cos"].shape[0])
            target_test_loss = target_res["test_cos"][random_indices[:mix_length]]
            target_test_loss = torch.tensor(target_test_loss).cpu().numpy()
            mix_test_loss = torch.tensor(target_res["mix_cos"]).cpu().numpy()
            mix_test_loss = np.concatenate([target_test_loss,mix_test_loss],axis=0)
            print('mix_test_loss shape:',mix_test_loss.shape)
            target_test_loss = mix_test_loss

    if attack_mode=="diff":
        target_train_loss=torch.tensor(target_res["tarin_diffs"]).cpu().numpy()
        if MODE=="test":
            target_test_loss=torch.tensor(target_res["test_diffs"]).cpu().numpy()
        elif MODE=="val":
            target_test_loss=torch.tensor(target_res["val_diffs"]).cpu().numpy()
    if attack_mode == 'loss':
        target_train_loss = -ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] ).cpu().numpy()
        if MODE=="test":
            target_test_loss=-ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] ).cpu().numpy()
        elif MODE=="val":
            target_test_loss=-ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] ).cpu().numpy()
        elif MODE == 'mix':
            random_indices = torch.randperm(target_res["test_res"]["logit"].shape[0])
            target_test_loss =-ce_loss_fn(target_res["test_res"]["logit"][random_indices[:mix_length]],\
                                            target_res["test_res"]["labels"][random_indices[:mix_length]])
            target_test_loss = torch.tensor(target_test_loss).cpu().numpy()
            mix_test_loss=-ce_loss_fn(target_res["mix_res"]["logit"] , target_res["mix_res"]["labels"] ).cpu().numpy()
            mix_test_loss = np.concatenate([target_test_loss,mix_test_loss],axis=0)
            print('mix_test_loss shape:',mix_test_loss.shape)
            target_test_loss = mix_test_loss

    shadow_train_losses=[]
    shadow_test_losses=[]
    if attack_mode=="cos":
        for i in shadow_res:
            shadow_train_losses.append( torch.tensor(i["tarin_cos"]).cpu().numpy() )
            if MODE=="val":
                shadow_test_losses.append(torch.tensor(i["val_cos"]).cpu().numpy() )
            elif MODE=="test":
                shadow_test_losses.append(torch.tensor(i["test_cos"]).cpu().numpy() )
            elif MODE =='mix':
                random_indices = torch.randperm(i["test_cos"].shape[0])
                shadow_test_loss = i["test_cos"][random_indices[:mix_length]]
                shadow_test_loss = torch.tensor(shadow_test_loss).cpu().numpy()
                mix_test_loss = torch.tensor(i["mix_cos"]).cpu().numpy()
                mix_test_loss = np.concatenate([shadow_test_loss,mix_test_loss],axis=0)
                print('mix_test_loss shape:',mix_test_loss.shape)
                shadow_test_losses.append(mix_test_loss)
    elif attack_mode=="diff":
        for i in shadow_res:
            shadow_train_losses.append( torch.tensor(i["tarin_diffs"]).cpu().numpy() )
            if MODE=="val":
                shadow_test_losses.append(torch.tensor(i["val_diffs"]).cpu().numpy() )
            elif MODE=="test":
                shadow_test_losses.append(torch.tensor(i["test_diffs"]).cpu().numpy() )
    elif attack_mode=="loss":
        for i in shadow_res:
            shadow_train_losses.append(-ce_loss_fn(i["train_res"]["logit"] , i["train_res"]["labels"]).cpu().numpy() )
            if MODE=="val":
                shadow_test_losses.append(-ce_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"]).cpu().numpy() )
            elif MODE=="test":
                shadow_test_losses.append(-ce_loss_fn(i["test_res"]["logit"], i["test_res"]["labels"]).cpu().numpy() )
            elif MODE == 'mix':
                random_indices = torch.randperm(i["test_res"]["logit"].shape[0])
                shadow_test_loss =-ce_loss_fn(i["test_res"]["logit"][random_indices[:mix_length]],\
                                                i["test_res"]["labels"][random_indices[:mix_length]])
                shadow_test_loss = torch.tensor(shadow_test_loss).cpu().numpy()
                mix_test_loss=-ce_loss_fn(i["mix_res"]["logit"] , i["mix_res"]["labels"] ).cpu().numpy()
                mix_test_loss = np.concatenate([shadow_test_loss,mix_test_loss],axis=0)
                print('mix_test_loss shape:',mix_test_loss.shape)
                shadow_test_losses.append(mix_test_loss)

    shadow_train_losses_stack=np.vstack( shadow_train_losses )
    shadow_test_losses_stack=np.vstack( shadow_test_losses )
    # print('shadow_train_losses_stack:',shadow_train_losses_stack.shape)

    ## 打印local model统计信息，观察结果
    print('mean 0 \t train:',target_train_loss.mean(axis=0),'\tvar:',target_train_loss.var(axis=0), ' \t test:', target_test_loss.mean(axis=0),'\tvar:',target_test_loss.var(axis=0))
    
    i=1
    for train_loss, test_loss in zip(shadow_train_losses, shadow_test_losses):
        print('mean',i, ' \t train:',train_loss.mean(axis=0),'\tvar:',train_loss.var(axis=0),' \t test:',test_loss.mean(axis=0), '\tvar:',test_loss.var(axis=0) )
        i+=1
    view_list = [0,1,2,3,4, 500,501,502,503,504, -5,-4,-3,-2,-1]
    print('########### Training samples: ############')
    print('Sample  ', end='')
    for j in view_list:
        print(f'{j}      ',' \t', end='')
    print('')
    ### 第0个client
    print('Client 0 ', end='')
    for j in view_list:
        view_score = '%.6f' % target_train_loss[j]
        print(f'{view_score} \t', end='')
    print('')
    ### 输出统计信息
    print('Mean ', end='')
    for j in view_list:
        view_score = '%.6f' % np.mean(shadow_train_losses_stack, axis=0)[j]
        print(f'{view_score} \t', end='')
    print('')
    print('Var  ', end='')
    for j in view_list:
        view_score = '%.6f' % np.var(shadow_train_losses_stack, axis=0)[j]
        print(f'{view_score} \t', end='')
    print('')
    
    ### 第1-9个client
    for i, train_loss in zip(range(1,K), shadow_train_losses):
        print(f'Client {i} ', end='')
        for j in view_list:
            view_score = '%.6f' % train_loss[j]
            print(f'{view_score} \t', end='')
        print('')
    print('')
    print('########### Testing samples: ############')
    ### 第0个client
    print('Client 0', end='')
    for j in view_list:
        view_score = '%.6f' % target_test_loss[j]
        print(f'{view_score} \t', end='')
    print('')

    print('Mean ', end='')
    for j in view_list:
        view_score = '%.6f' % np.mean(shadow_test_losses_stack, axis=0)[j]
        print(f'{view_score} \t', end='')
    print('')
    print('Var  ', end='')
    for j in view_list:
        view_score = '%.6f' % np.var(shadow_test_losses_stack, axis=0)[j]
        print(f'{view_score} \t', end='')
    print('')
    ### 第1-9个client
    for i, test_loss in zip(range(1,K ), shadow_test_losses):
        print(f'Client {i} ', end='')
        for j in view_list:
            view_score = '%.6f' % test_loss[j]
            print(f'{view_score} \t', end='')
        print('')
    print('select_mode',select_mode, type(select_mode))
    print('select_method',select_method)
    print('attack_mode:',attack_mode)

    if select_mode == 1 and attack_mode =='cos':
        # print('***********first in*************')
        tmps=[]
        means=[]
        client_ids=[]
        
        if select_method == 'outlier':
            # shadow_mdm_stack = np.vstack(shadow_train_losses_stack, shadow_test_losses_stack)
            train_mu_out=np.zeros_like(shadow_train_losses_stack.mean(axis=0))
            train_var_out=np.zeros_like(shadow_train_losses_stack.var(axis=0)+1e-8)
            print('**************',train_mu_out.shape)
            test_mu_out=np.zeros_like(shadow_test_losses_stack.mean(axis=0))
            test_var_out=np.zeros_like(shadow_test_losses_stack.var(axis=0)+1e-8)

            for j in range(0,shadow_train_losses_stack.shape[1]):
                mask = shadow_train_losses_stack[:,j] < shadow_train_losses_stack[:,j].mean(axis=0) + 3*shadow_train_losses_stack[:,j].std(axis=0)
                sel_mdm = shadow_train_losses_stack[:,j][mask]
                if j %2000==0:
                    print(' train outlier view:')
                    print(shadow_train_losses_stack[:,j])
                    print(target_train_loss[j])
                    print('sel_mdm.shape', sel_mdm.shape)
                if sel_mdm.shape[0]==0:
                    if j % 50 == 0:
                        print('outlier view:')
                        print('mask:',mask)
                        print(shadow_train_losses_stack[:,j])
                        print(target_train_loss[j])
                    sel_mdm=np.array([np.min(shadow_train_losses_stack[:,j])])
                train_mu_out[j] = np.mean(sel_mdm, axis=0)
                train_var_out[j] = np.var(sel_mdm, axis=0)+1e-8
            
            for j in range(0,shadow_test_losses_stack.shape[1]):
                mask = shadow_test_losses_stack[:,j] < shadow_test_losses_stack[:,j].mean(axis=0) + 3*shadow_test_losses_stack[:,j].std(axis=0)
                sel_mdm = shadow_test_losses_stack[:,j][mask]
                if j % 10==0:
                    print('outlier view:')
                    print(shadow_test_losses_stack[:,j])
                    print(target_test_loss[j])
                    print(mask)
                    print('sel_mdm.shape', sel_mdm.shape)
                # sel_mdm = np.sort(shadow_test_losses_stack[:,j])[2:3+SHADOW_NUM]
                # sel_mdm = shadow_test_losses_stack[:,j][mask]
                
                if j %2000==0:
                    print('test outlier view:')
                    print(shadow_test_losses_stack[:,j])
                    print(target_test_loss[j])
                    print('sel_mdm.shape', sel_mdm.shape)
                if sel_mdm.shape[0]==0:
                    if j % 50==0:
                        print('outlier view:')
                        print(shadow_test_losses_stack[:,j])
                        print(target_test_loss[j])
                        print('sel_mdm.shape', sel_mdm.shape)
                    sel_mdm=np.array([np.min(shadow_test_losses_stack[:,j])])
                test_mu_out[j] = np.mean(sel_mdm, axis=0)
                test_var_out[j] = np.var(sel_mdm, axis=0)+1e-8    

    ## 计算均值和方差，以备分布估计
    if attack_mode != 'cos'or select_mode == 0 or (select_method != 'mean_per' and select_method != 'outlier'):
        train_mu_out=shadow_train_losses_stack.mean(axis=0)
        train_var_out=shadow_train_losses_stack.var(axis=0)+1e-8

        test_mu_out=shadow_test_losses_stack.mean(axis=0)
        test_var_out=shadow_test_losses_stack.var(axis=0)+1e-8

    if epch % 50 ==0:
        print('target_train_loss:', target_train_loss[0:10])
        print('train_mu_out:', train_mu_out[0:10])

        print('target_test_loss:', target_test_loss[0:10])
        print('test_mu_out:', test_mu_out[0:10])

    # 计算概率密度
    # train_l_out=scipy.stats.norm.cdf(target_train_loss,train_mu_out,np.sqrt(train_var_out))
    # # train_l_out=1-scipy.stats.norm.cdf(target_train_loss,test_mu_out,np.sqrt(test_var_out))
    # train_mu_out = np.ones_like(target_train_loss) * test_mu_out.mean(axis=0)
    # train_var_out = np.ones_like(target_train_loss) * test_var_out.mean(axis=0)

    train_l_out=scipy.stats.norm.cdf(target_train_loss,train_mu_out,np.sqrt(train_var_out))
    test_l_out=scipy.stats.norm.cdf(target_test_loss,test_mu_out,np.sqrt(test_var_out))
    print('var of train:', np.sqrt(train_var_out).mean(axis=0),'var of test:', np.sqrt(test_var_out).mean(axis=0))

    print("attack_mode:",attack_mode)

    print("mean of train_l_out:",train_l_out.mean(axis=0),"var of train_l_out:",train_l_out.var(axis=0))
    print("mean of test_l_out:",test_l_out.mean(axis=0),"var of test_l_out:",test_l_out.var(axis=0))
    print(test_l_out.shape)
    print('Checking traing sample score:')
    print(train_l_out[0:5])
    print(train_l_out[100:105])
    print(train_l_out[500:505])
    print(train_l_out[1500:1505])
    print('Checking test sample score:')
    print(test_l_out[0:5])
    print(test_l_out[100:105])
    print(test_l_out[500:505])
    print(test_l_out[1500:1505])
    # print()
    outlier_indexs = np.array(np.where(test_l_out > 0.8))
    print("###################################")
    print('############# outlier view, num:', outlier_indexs.shape)
    print("###################################")
    print(test_l_out[outlier_indexs[0,0:5]])
    print(target_test_loss[outlier_indexs[0,0:5]])
    print(shadow_test_losses_stack[:,[outlier_indexs[0,0:5]]] )
    print('____________________________')
    print(test_l_out[outlier_indexs[0,20:25]])
    print(target_test_loss[outlier_indexs[0,20:25]])
    print(shadow_test_losses_stack[:,[outlier_indexs[0,20:25]]] )
    print('____________________________')
    print(test_l_out[outlier_indexs[0,50:55]])
    print(target_test_loss[outlier_indexs[0,50:55]])
    print(shadow_test_losses_stack[:,[outlier_indexs[0,50:55]]] )
    print('____________________________')
    print(test_l_out[outlier_indexs[0,100:105]])
    print(target_test_loss[outlier_indexs[0,100:105]])
    print(shadow_test_losses_stack[:,[outlier_indexs[0,100:105]]] )

    print('mem outlier view')
    outlier_indexs = np.array(np.where(train_l_out <0.5))
    print('outlier num:', outlier_indexs.shape)
    print(train_l_out[outlier_indexs[0,0:5]])
    print(target_train_loss[outlier_indexs[0,0:5]])
    print(shadow_train_losses_stack[:,[outlier_indexs[0,0:5]]] )
    print('____________________________')
    print(train_l_out[outlier_indexs[0,20:25]])
    print(target_train_loss[outlier_indexs[0,20:25]])
    print(shadow_train_losses_stack[:,[outlier_indexs[0,20:25]]] )

    auc,log_auc,tprs=plot_auc("lira",torch.tensor(test_l_out),torch.tensor(train_l_out),epch)
    # auc,log_auc,tprs=plot_auc("lira", (torch.tensor((test_mu_out))), (torch.tensor((train_mu_out))),epch)
    if epch % 10 ==0:
        print("__"*10,"lira_attack")
        print(f"tprs:{tprs}",log_auc)

    return accs,tprs,auc,log_auc,(train_l_out,test_l_out)

def cos_attack(f,K,epch,attack_mode,extract_fn=None):

    accs=[]
    target_res=torch.load(f.format(0,epch))
    tprs=None
    print(attack_mode)

    if attack_mode =="cosine attack":
        if MODE=="test":
            val_liratios=target_res['test_cos']
        elif MODE=="val":
            val_liratios=target_res['val_cos']
        elif MODE=='mix':
            random_indices = torch.randperm(target_res["test_cos"].shape[0])
            val_liratios = target_res["test_cos"][random_indices[:mix_length]]
            val_liratios = torch.tensor(val_liratios)
            mix_test_loss = torch.tensor(target_res["mix_cos"])
            mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
            # print('mix_test_loss shape:',mix_test_loss.shape)
            val_liratios = mix_test_loss
        # print(val_liratios)

        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_cos']
        train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("cos_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
  
        if epch % 50 ==0:
            print("__"*10,"cos_attack")
            print(f"tprs:{tprs}",log_auc)
            # print("test_acc:",target_res[taret_idx])
            print("__"*10,)

    elif attack_mode =="grad diff":
        if MODE=="test":
            val_liratios=target_res['test_diffs']
        elif MODE=="val":
            val_liratios=target_res['val_diffs']
        elif MODE=='mix':
            random_indices = torch.randperm(target_res["test_diffs"].shape[0])
            val_liratios = target_res["test_diffs"][random_indices[:mix_length]]
            val_liratios = torch.tensor(val_liratios)
            mix_test_loss = torch.tensor(target_res["mix_diffs"])
            mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
            # print('mix_test_loss shape:',mix_test_loss.shape)
            val_liratios = mix_test_loss
        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_diffs']
        train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("diff_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)   
        if epch % 50 ==0:
            print("__"*10,"diff_attack")
            print(f"tprs:{tprs}",log_auc)
            # print("test_acc:",target_res[taret_idx])
            print("__"*10,)
    elif attack_mode =="grad norm":
        if MODE=="test":
            val_liratios=target_res['test_grad_norm']
        elif MODE=="val":
            val_liratios=target_res['val_grad_norm']
        elif MODE=='mix':
            random_indices = torch.randperm(target_res["test_grad_norm"].shape[0])
            val_liratios = target_res["test_grad_norm"][random_indices[:mix_length]]
            val_liratios = -torch.tensor(val_liratios)
            mix_test_loss = -torch.tensor(target_res["mix_grad_norm"])
            mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
            # print('mix_test_loss shape:',mix_test_loss.shape)
            val_liratios = mix_test_loss
        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_grad_norm']
        train_liratios=-np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("grad_norm_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch) 
        if epch % 50 ==0:
            print("__"*10,"grad_norm_attack")
            print(f"tprs:{tprs}",log_auc)
            # print("test_acc:",target_res[taret_idx])
            print("__"*10,)
    elif attack_mode =="loss based":
        if MODE=="test":
            val_liratios=-ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
        elif MODE=="val":
            val_liratios=-ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] )
        elif MODE =='mix':
            random_indices = torch.randperm(target_res["test_res"]["logit"].shape[0])
            val_liratios =-ce_loss_fn(target_res["test_res"]["logit"][random_indices[:mix_length]],\
                                            target_res["test_res"]["labels"][random_indices[:mix_length]])
            mix_test_loss=-ce_loss_fn(target_res["mix_res"]["logit"] , target_res["mix_res"]["labels"] ).cpu().numpy()
            mix_test_loss = np.concatenate([val_liratios,mix_test_loss],axis=0)
            # print('mix_test_loss shape:',mix_test_loss.shape)
            val_liratios = mix_test_loss

        # val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=-ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
        # train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("loss_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)    
        if epch % 50 ==0:
            print("__"*10,"loss_attack")
            print(f"tprs:{tprs}",log_auc)
            # print("test_acc:",target_res[taret_idx])
            print("__"*10,)

    return accs,tprs,auc,log_auc,(train_liratios, val_liratios)

def fig_out(x_axis_data, MAX_K,defence,seed,log_path, d,avg_d=None,single_score=None, other_scores=None,accs=None): 
    colors={
        "cosine attack":"r",
        "grad diff":"g",
        "loss based":"b",
        "grad norm":(242/256, 159/256, 5/256),
        "lira":"y",
        "log_lira":"k",
        "lira_loss":'purple'
            }
    labels_per_epoch = {
        "cosine attack":"Grad-Cosine",
        "grad diff":"Grad-Diff",
        "loss based":"Blackbox-Loss",
        "grad norm":"Grad-Norm"
    }
    labels_temporal = {
        "cosine attack":"Avg-Cosine",
        "loss based":"Loss-Series",
        "lira":"FedMIA-II",
        "lira_loss":"FedMIA-I"
    }
    fig = plt.figure(figsize=(6.5, 6.5), dpi=200)
    fig.subplots_adjust(top=0.91,
                        bottom=0.160,
                        left=0.180,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    for k in labels_per_epoch.keys():
        print(k, d[k])
        plt.plot(x_axis_data[0:len(d[k])], d[k], linewidth=1, label=labels_per_epoch[k], color=colors[k])
    # plt.plot(x_axis_data, common_score,'bo-', linewidth=1, color='#2E8B57', label=r'Baseline')
    plt.legend(loc=3)  

    plt.xlim(-2, 305)
    my_x_ticks = np.arange(0, 302, 50)
    plt.xticks(my_x_ticks,size=14)
    if avg_d:
        for k in labels_temporal.keys():
            if avg_d[k]:    
                plt.hlines([avg_d[k]["0.001"]],xmin=0,xmax=300,label=labels_temporal[k],color=colors[k])

    plt.legend(prop={'size': 10})
    plt.xlabel('Epoch',fontsize=14,fontdict={'size': 14})  # x_label
    plt.ylabel('TPR@FPR=0.001',fontsize=14,fontdict={'size': 14})  # y_label
    plt.grid(axis='both')

    pdf_path=PATH.split("/")[0:-1]
    pdf_path="/".join(pdf_path)+f"/attack_fig_{select_mode}_{select_method}_n{SHADOW_NUM}_s{SEED}.pdf"
    
    # pdf_path="/".join(pdf_path)+"/attack9_val_mode_positive_plus.pdf"
    # attack9_val_mode_positive_plus_select_mean_<<.pdf
    print('fig saved in', pdf_path)
    plt.savefig(pdf_path)

    # print("log_path0:",log_path)
    # log_path=log_path+f"/def{defence}2_0.85_k{MAX_K}_{seed}_attack.log"
    log_path=PATH.split("/")[0:-1]
    log_path="/".join(log_path)+f"/attack_score_{select_mode}_{select_method}_n{SHADOW_NUM}_s{SEED}.log"
    # print("log_path:",log_path)
    with open(log_path,"w") as f:
        json.dump({"avg_d":avg_d,"single_score":single_score,"other_scores":other_scores,"accs":accs},f, indent=4)
    # assert 0

@ torch.no_grad()
def attack_comparison(p,log_path, save_dir, epochs, MAX_K, defence,seed):
    """
    Summary of the Correspondence between attack methods in paper and scores in codea:
    summary_dict={
    'Blackbox-Loss': scores['loss based'],
    'Grad-Cosine': scores['cosine attack'],
    'Grad-Diff': scores['grad diff'],
    'Grad-Norm': scores['grad norm'],

    'Loss-Series': avg_scores["loss based"],
    'Avg-Cosine': avg_scores["cosine attack"],

    'FedMIA-I': avg_scores["lira_loss"],
    'FedMIA-II': avg_scores["lira"]
    }
    """

    final_acc=lira_attack_ldh_cosine(p,epochs[-1],MAX_K, save_dir, extract_fn=extract_hinge_loss)[0]

    lira_scores=[]
    lira_loss_scores=[]
    common_scores=[]
    other_scores={}

    ## 记录TPR@FPR=0.01
    scores={k:[] for k in attack_modes}
    scores["lira"]=[]
    scores["lira_loss"]=[]
    ## 记录所有epoch的TPR@FPR=0.01中最大的
    single_score={k:0 for k in attack_modes}
    single_score["lira"]=0
    single_score["lira_loss"]=0
    ## 记录每轮 lira的mem和non-mem的cdf, 即(train_l_out,test_l_out) 
    reses_lira=[]
    reses_lira_loss=[]
    ## 记录其他attack mode的 (val_liratios,train_liratios)
    reses_common={k:[] for k in attack_modes}
    ## 记录所有轮数cdf avg后AUC攻击的TPR得分
    avg_scores={k:None for k in attack_modes}
    avg_scores["lira"]=None
    avg_scores["lira_loss"]=None

    auc_dict={k:[] for k in attack_modes}
    auc_dict["lira"]=[]
    auc_dict["lira_loss"]=[]

    for epch in epochs:
        # try:
        lira_score=lira_attack_ldh_cosine(p,epch,MAX_K,save_dir, extract_fn=extract_hinge_loss) 
        lira_loss_score=lira_attack_ldh_cosine(p,epch,MAX_K,save_dir, extract_fn=extract_hinge_loss,attack_mode='loss') 
            
            # the above function retruns: accs, tprs, auc, log_auc, (train_l_out,test_l_out) 
            ## log_auc: 基于log_lira所得的auc
        # except ValueError:
        #     print("ValueError")
        #     continue
        scores["lira"].append(lira_score[1]['0.001'])
        scores["lira_loss"].append(lira_loss_score[1]['0.001'])
        auc_dict["lira"].append(lira_score[2])
        auc_dict["lira_loss"].append(lira_loss_score[2])



        # lira_score=lira_attack(p,epch,K=9,extract_fn=extract_hinge_loss)
        for attack_mode in attack_modes:
            common_score=cos_attack(p,0,epch,attack_mode,extract_fn=extract_hinge_loss) 
            # the above function return:  accs, tprs, auc, log_auc, (val_liratios,train_liratios)
            reses_common[attack_mode].append(common_score[-1])
            scores[attack_mode].append(common_score[1]['0.001'])
            auc_dict[attack_mode].append(common_score[2])
            print('____________________',attack_mode)
            print(common_score[1])
            if epch ==200 and attack_mode=="loss based":
                other_scores["loss_single_epch_score"]=common_score[1] # tpr
                other_scores["loss_single_auc"]=[common_score[2],common_score[3]] # tpr, auc

        lira_scores.append(lira_score[1]['0.001'])
        lira_loss_scores.append(lira_loss_score[1]['0.001'])
        common_scores.append(common_score[1]['0.001']) # 为最后一个loss based的common_score, 但似乎这个list没啥用

        reses_lira.append(lira_score[-1]) # 当下epoch的 (train_l_out,test_l_out) 
        reses_lira_loss.append(lira_loss_score[-1])

    for attack_mode in attack_modes:
        sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
        single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
        single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]


    for attack_mode in ['lira', 'lira_loss']:
        sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
        single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
        single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]

    for attack_mode in attack_modes:
        # print('len(scores[attack_mode]): ',len(scores[attack_mode]))  30
        single_score[f'200 {attack_mode}']=(scores[attack_mode][int(epochs[-1]/20)])
        single_score[f'200 single_{attack_mode}_auc'] = auc_dict[attack_mode][int(epochs[-1]/20)]
    for attack_mode in ['lira', 'lira_loss']:
        single_score[f'200 {attack_mode}']=(scores[attack_mode][int(epochs[-1]/20)])
        single_score[f'200 single_{attack_mode}_auc'] = auc_dict[attack_mode][int(epochs[-1]/20)]


    print('------------ ----------------- -------------  ')
    print('------------ Sequential attack -------------  ')
    print('------------ ----------------- -------------  ')

    reses=reses_lira
    train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)

    print('avged train_score.shape:',train_score.shape)
    print('******************************************')
    print('***********check avg lira attack:*********')
    print('******************************************')
    print(np.mean(train_score),np.var(train_score), train_score.shape)
    print(np.mean(test_score), np.var(test_score), test_score.shape)


    auc,log_auc,tprs=plot_auc("averaged_lira",torch.tensor(test_score),torch.tensor(train_score),999)
    print(f"averaged_lira tprs:{tprs} \n auc:{auc}")
    avg_scores["lira"]=tprs
    other_scores["lira_auc"]=[auc,log_auc]
    print("success!")

    reses=reses_lira_loss
    train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    #print(train_score.shape)
    # assert 0
    auc,log_auc,tprs=plot_auc("averaged_lira_loss",torch.tensor(test_score),torch.tensor(train_score),999)
    print(f"averaged_lira_loss tprs:{tprs} \n auc:{auc}")
    avg_scores["lira_loss"]=tprs
    other_scores["lira_loss_auc"]=[auc,log_auc]
    print("success!")

    reses=reses_lira
    train_score=np.nanmean(1-np.log(np.vstack([ i[0].reshape(1,-1) for i in reses])),axis=0)
    test_score=np.nanmean(1-np.log(np.vstack([ i[1].reshape(1,-1) for i in reses])),axis=0)
    print(train_score.min(),train_score.max())
    train_score=train_score[~(np.isnan(train_score))]
    test_score=test_score[~(np.isnan(test_score))]
    train_score[(np.isinf(train_score))]=-1e10
    test_score[(np.isinf(test_score))]=-1e10
    print(train_score.min(),train_score.max())
    print(train_score.shape)
    print('***********check avg log_lira attack:')
    print(np.mean(train_score),np.var(train_score), train_score.shape)
    print(np.mean(test_score), np.var(test_score), test_score.shape)
    auc,log_auc,tprs=plot_auc("averaged_log_lira",torch.tensor(test_score),torch.tensor(train_score),999)
    print(f"averaged_log_lira tprs:{tprs} \n auc:{auc}")
    avg_scores["log_lira"]=tprs
    other_scores["log_lira_auc"]=[auc,log_auc]
    print("success!")

    reses=reses_common["cosine attack"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    print('***********check avg cos attack:')
    print(np.mean(train_score),np.var(train_score), train_score.shape)
    print(np.mean(test_score), np.var(test_score), test_score.shape)

    auc,log_auc,tprs=plot_auc("cosine attack",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["cosine attack"]=tprs
    other_scores["cos_attack_auc"]=[auc,log_auc]
    print(f"averaged cosine attack tprs:{tprs} \n auc:{auc}")
    print("success!")

    reses=reses_common["grad diff"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    print('***********check avg grad diff attack:')
    print(np.mean(train_score),np.var(train_score), train_score.shape)
    print(np.mean(test_score), np.var(test_score), test_score.shape)
    auc,log_auc,tprs=plot_auc("averaged_diff",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["grad diff"]=tprs
    other_scores["grad_diff_auc"]=[auc,log_auc]
    print(f"averaged_diff tprs:{tprs} \n auc:{auc}")
    print("success!")

    reses=reses_common["grad norm"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=-np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=-np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    auc,log_auc,tprs=plot_auc("averaged_norm",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["grad norm"]=tprs
    other_scores["grad_norm_auc"]=[auc,log_auc]
    print(f"tprs:{tprs}")
    print("success!")

    reses=reses_common["loss based"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    print('***********check avg loss based attack:')
    print(np.mean(train_score),np.var(train_score), train_score.shape)
    print(np.mean(test_score), np.var(test_score), test_score.shape)

    auc,log_auc,tprs=plot_auc("averaged_loss",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["loss based"]=tprs
    other_scores["loss_based_auc"]=[auc,log_auc]
    print(f"averaged_loss tprs:{tprs} \n auc:{auc}")
    print("success!")

    fig_out(epochs,MAX_K, defence,seed,log_path,scores,avg_scores,single_score, other_scores,final_acc)


def main(argv):
    global MODE, attack_modes, PATH, p_folder, device, select_mode, select_method, SHADOW_NUM, SEED, mix_length
    global SAVE_DIR
    
    attack_modes=["cosine attack","grad diff","loss based","grad norm"]
    epochs=list(range(10,int(argv[2])+1,10))
    p_folder=argv[1]  
    PATH=argv[1]
    device=argv[3] 
    MODE='mix'
    SEED = int(argv[4])
    MAX_K=10
    
    for root, dirs, files in os.walk(p_folder, topdown=False):
        for name in dirs:
            if  root!=p_folder:#or 's1' not in name or model not in name: #or 's5' not in name:
                continue
            else: 
                PATH=os.path.join(root, name)
                PATH+="/client_{}_losses_epoch{}.pkl"
                MAX_K=int(name.split("_K")[1].split("_")[0])
                model=name.split("_")[3]
                defence=name.split("_")[-5].strip('def').strip('0.0')
                seed=name.split("_")[-1]
                save_dir=p_folder + '/'+name
                SAVE_DIR = save_dir

                if 'iid$1' in name:
                    select_mode = 0
                    select_method='none'
                    SHADOW_NUM = 9
                else:
                    select_mode = 1
                    select_method ='outlier'
                    SHADOW_NUM = 4
                print(os.path.join(root, name))

                print('MODE\tattack_modes\tPATH\tp_folder\tselect_mode\tselect_method\tSHADOW_NUM\tSEED')
                print(f'{MODE}\t{attack_modes}\t{PATH}\t{p_folder}\t{select_mode}\t{select_method}\t{SHADOW_NUM}\t{SEED}')
                print("name:",name)

                if 'cifar100' in name:
                    mix_length=int(10000/MAX_K)
                elif 'dermnet' in name:
                    mix_length=4000
                if model== "alexnet":
                    log_path="logs/log_alex"
                else:
                    log_path="logs/log_res"
                # print(MAX_K,PATH)
                try:
                    attack_comparison(PATH, log_path, save_dir, epochs, MAX_K, defence,seed)
                    print("success!")
                    
                except IOError:
                    print("error:",MAX_K,PATH)
                    pass
            rewrite_print(os.path.join(root, name))

## Override the print function to save output information to the file
# Save the original print function
rewrite_print = print
# Define a new print function
def print(*arg, end=None):
    global SAVE_DIR
    file_path = SAVE_DIR + f'/attack_select_{select_mode}_{select_method}20_{MODE}_n{SHADOW_NUM}_s{SEED}_running.log'
    if end == None:
        rewrite_print(*arg, file=open(file_path, "a", encoding="utf-8"))
    else:
        rewrite_print(*arg, end='', file=open(file_path, "a", encoding="utf-8"))

if __name__ == "__main__":
    main(sys.argv)

