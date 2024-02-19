# %%
import sys
import os
import numpy as np
import torch
from collections import ChainMap
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import scipy
import time
import json

import warnings
warnings.filterwarnings('ignore')

 #.cuda(device)
# %%

def extract_loss(i):
    val_dict={}
    #for j,k in zip(i['val_index'],i['val_losses']):
    for j,k in zip(i['val_index'],i['val_res']['loss']):
        if j in val_dict:
            # assert 0
            val_dict[j].append(k)
        else:
            val_dict[j]=[k]
    train_dict={}
    #for j,k in zip(i['train_index'],i['train_losses']):
    for j,k in zip(i['train_index'],i['train_res']['loss']):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j]=[k]
    return (val_dict,train_dict)

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

def extract_hinge_loss_ldh(i):
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


def extract_loss2logit(i):
    val_dict={}
    val_index=i["val_index"]
    val_hinge_index=get_logit(i["val_res"]["loss"] )
    for j,k in zip(val_index,val_hinge_index):
        if j in val_dict:
            val_dict[j].append(k)
        else:
            val_dict[j]=[k]
    train_dict={}
    train_index=i["train_index"]
    train_hinge_index=get_logit(i["train_res"]["loss"] )
    for j,k in zip(train_index,train_hinge_index):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j]=[k]
    return (val_dict,train_dict)

def merge_dicts(lst):
    new_dict={}
    for d in lst:
        for k in d.keys():
            if k in new_dict:
                new_dict[k].extend(d[k])
            else:
                new_dict[k]= d[k] 
    #print("merge_dicts:", len(new_dict))
    return new_dict

def get_logit(l):
    return np.array(l)

def plot_auc(name,target_val_score,target_train_score,epoch): 
    global fpr, tpr
    fpr, tpr, thresholds = metrics.roc_curve(torch.cat( [torch.zeros_like(target_val_score),torch.ones_like(target_train_score)] ).cpu().numpy(), torch.cat([target_val_score,target_train_score]).cpu().numpy())
    auc=metrics.auc(fpr, tpr)
    log_tpr,log_fpr=np.log10(tpr),np.log10(fpr)
    log_tpr[log_tpr<-5]=-5
    log_fpr[log_fpr<-5]=-5
    log_fpr=(log_fpr+5)/5.0
    log_tpr=(log_tpr+5)/5.0
    log_auc=metrics.auc( log_fpr,log_tpr )

    tprs={}
    for fpr_thres in [0.1,0.02,0.01,0.001,0.0001]:
        tpr_index = np.sum(fpr<fpr_thres)
        tprs[str(fpr_thres)]=tpr[tpr_index-1]
    return auc,log_auc,tprs


def get_logit(l):
    p=np.exp(-np.array(l))
    return np.log(p/(1-p+1e-15))


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


def lira_attack_testidx(f,K,extract_fn=None):
    accs=[]
    training_res=[]
    for i in range(K):
        training_res.append(torch.load(f.format(i)))
        accs.append(training_res[-1]["test_acc"])


    losses_dict=[ extract_fn(i) for i in training_res]
    # losses_dict=[ extract_loss2logit(i) for i in training_res]
    train_dict=[ i[1] for i in losses_dict]
    #print("len(train_dict):",len(train_dict))
    val_dict=[i[0] for i in losses_dict]
    train_dict=merge_dicts(train_dict)
    val_dict=merge_dicts(val_dict)
    logit_train=[]
    logit_val=[]
    #print("len(train_dict.keys()):",len(train_dict.keys()))

    #Eliminate data indexes of target model
    lira_target=torch.load(f.format(K))

    target_val,target_train=extract_fn(lira_target)
    # target_val,target_train=extract_loss2logit(lira_target)
    target_val= {key: value for key, value in target_val.items()}
    target_train= {key: value for key, value in target_train.items()}

    lira_train_idxs=list(range(50000))
    for idx in target_train.keys():
        lira_train_idxs.remove(idx)
    #print("len(lira_train_idxs)",len(lira_train_idxs))

    for k in lira_train_idxs:
        logit_train.append(train_dict[k])

    for k in range(50000):
        # if (k in target_train.keys())!= True:
        logit_val.append(val_dict[k])
        if k in target_train.keys():
            del logit_val[k][8]

    logit_train=np.array(logit_train)
    print(len(logit_val))
    logit_val=np.array(logit_val)

    mu_in=logit_train.mean(axis=1)
    mu_out=logit_val.mean(axis=1)

    var_in=logit_train.var() # global variance
    var_out=logit_val.var()

    lira_target=torch.load(f.format(K))

    target_val,target_train=extract_fn(lira_target)
    # target_val,target_train=extract_loss2logit(lira_target)
    target_val= {key: value for key, value in target_val.items()}
    target_train= {key: value for key, value in target_train.items()}
    logits_new=[]
    for k in range(50000):
        if k in target_val.keys():
            logits_new.append(target_val[k][0])
        elif k in target_train.keys():
            logits_new.append(target_train[k][0])
        else:
            assert 0
    logits_new=np.array(logits_new)

    #print("new:",len(logits_new))
    #l_in,l_out=liratio(mu_in,mu_out,var_in,var_out,logits_new)
    l_out=liratio(mu_in,mu_out,var_in,var_out,logits_new)

    liratios=1-l_out
    val_liratios=np.array([liratios[i] for i in target_val.keys()])
    train_liratios=np.array([liratios[i] for i in target_train.keys()])
    auc,log_auc,tprs=plot_auc(torch.tensor(val_liratios),torch.tensor(train_liratios))
    #accs.append(lira_target["test_acc"])
    #print(f"tprs:{tprs}")
    #print("test_acc:",lira_target["test_acc"])
    return accs,tprs,auc,log_auc,liratios


def lira_attack_ldh(f,epch,K,extract_fn=None):
    accs=[]
    training_res=[]
    for i in range(K):
        training_res.append(torch.load(f.format(i,epch)))
        accs.append(training_res[-1]["test_acc"])
    
    target_idx=0
    target_res=training_res[target_idx]
    shadow_res=training_res[target_idx+1:]

    target_train_loss=hinge_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
    if MODE=="test":
        target_test_loss=hinge_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
    elif MODE=="val":
        target_test_loss=hinge_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] )

    shadow_train_losses=[]
    shadow_test_losses=[]
    for i in shadow_res:
        shadow_train_losses.append( hinge_loss_fn(i["train_res"]["logit"] , i["train_res"]["labels"] ).reshape(1,-1) )
        if MODE=="val":
            shadow_test_losses.append(hinge_loss_fn(i["val_res"]["logit"] , i["val_res"]["labels"] ).reshape(1,-1) )
        elif MODE=="test":
            shadow_test_losses.append(hinge_loss_fn(i["test_res"]["logit"] , i["test_res"]["labels"] ).reshape(1,-1) )
    shadow_train_losses=np.vstack( shadow_train_losses )
    shadow_test_losses=np.vstack( shadow_test_losses )

    train_mu_out=shadow_train_losses.mean(axis=0)
    train_var_out=shadow_train_losses.var(axis=0)+1e-8
    # train_var_out=shadow_train_losses.var()+1e-8

    test_mu_out=shadow_test_losses.mean(axis=0)
    test_var_out=shadow_test_losses.var(axis=0)+1e-8
    # test_var_out=shadow_test_losses.var()+1e-8
    print(target_train_loss.shape,train_mu_out.shape,np.sqrt(train_var_out).shape)
    print(target_test_loss.shape,test_mu_out.shape,np.sqrt(test_var_out).shape)
    train_l_out=scipy.stats.norm.cdf(target_train_loss,train_mu_out,np.sqrt(train_var_out))
    test_l_out=scipy.stats.norm.cdf(target_test_loss,test_mu_out,np.sqrt(test_var_out))
    # assert 0

    # test_l_out=liratio(None,test_mu_out,None,test_var_out,target_test_loss)

    auc,log_auc,tprs=plot_auc("lira",torch.tensor(test_l_out),torch.tensor(train_l_out),epch)

    print("__"*10,"lira")
    print(f"tprs:{tprs}")
    print("test_acc:",accs[target_idx])
    print("__"*10,)

    return accs,tprs,auc,log_auc,(train_l_out,test_l_out)

def lira_attack_ldh_cosine(f,epch,K,extract_fn=None,attack_mode="cos"):
    accs=[]
    training_res=[]
    for i in range(K):
        print(i,epch)
        # training_res.append(torch.load(f.format(i,epch),map_location=lambda storage, loc: storage))
        training_res.append(torch.load(f.format(i,epch)))
        accs.append(training_res[-1]["test_acc"])
    
    target_idx=0
    target_res=training_res[target_idx]
    shadow_res=training_res[target_idx+1:]
    #print(target_res["tarin_cos"])
    if attack_mode=="cos":
        target_train_loss=torch.tensor(target_res["tarin_cos"]).cpu().numpy()
        if MODE=="test":
            target_test_loss=torch.tensor(target_res["test_cos"]).cpu().numpy()
        elif MODE=="val":
            target_test_loss=torch.tensor(target_res["val_cos"]).cpu().numpy()
    if attack_mode=="diff":
        target_train_loss=torch.tensor(target_res["tarin_diffs"]).cpu().numpy()
        if MODE=="test":
            target_test_loss=torch.tensor(target_res["test_diffs"]).cpu().numpy()
        elif MODE=="val":
            target_test_loss=torch.tensor(target_res["val_diffs"]).cpu().numpy()

    shadow_train_losses=[]
    shadow_test_losses=[]
    if attack_mode=="cos":
        for i in shadow_res:
            shadow_train_losses.append( torch.tensor(i["tarin_cos"]).cpu().numpy() )
            if MODE=="val":
                shadow_test_losses.append(torch.tensor(i["val_cos"]).cpu().numpy() )
            elif MODE=="test":
                shadow_test_losses.append(torch.tensor(i["test_cos"]).cpu().numpy() )
    if attack_mode=="diff":
        for i in shadow_res:
            shadow_train_losses.append( torch.tensor(i["tarin_diffs"]).cpu().numpy() )
            if MODE=="val":
                shadow_test_losses.append(torch.tensor(i["val_diffs"]).cpu().numpy() )
            elif MODE=="test":
                shadow_test_losses.append(torch.tensor(i["test_diffs"]).cpu().numpy() )

    shadow_train_losses=np.vstack( shadow_train_losses )
    shadow_test_losses=np.vstack( shadow_test_losses )

    train_mu_out=shadow_train_losses.mean(axis=0)
    train_var_out=shadow_train_losses.var(axis=0)+1e-8
    # train_var_out=shadow_train_losses.var()+1e-8

    test_mu_out=shadow_test_losses.mean(axis=0)
    test_var_out=shadow_test_losses.var(axis=0)+1e-8
    # test_var_out=shadow_test_losses.var()+1e-8

    train_l_out=1-scipy.stats.norm.cdf(target_train_loss,train_mu_out,np.sqrt(train_var_out))
    test_l_out=1-scipy.stats.norm.cdf(target_test_loss,test_mu_out,np.sqrt(test_var_out))

    auc,log_auc,tprs=plot_auc("lira",torch.tensor(test_l_out),torch.tensor(train_l_out),epch)

    return accs,tprs,auc,log_auc,(train_l_out,test_l_out)

def lira_attack(f,epch,K,extract_fn=None):
    accs=[]
    training_res=[]
    for i in range(K):
        training_res.append(torch.load(f.format(i,epch)))
        accs.append(training_res[-1]["test_acc"])

    losses_dict=[ extract_fn(i) for i in training_res]
    # losses_dict=[ extract_loss2logit(i) for i in training_res]
    train_dict=[ i[1] for i in losses_dict]
    val_dict=[i[0] for i in losses_dict]

    train_dict=merge_dicts(train_dict)
    val_dict=merge_dicts(val_dict)
    logit_train=[]
    logit_val=[]
    #print("len(train_dict.keys()):",len(train_dict.keys()))

    target_idx=0
    lira_target=torch.load(f.format(target_idx,epch))

    target_val,target_train=extract_fn(lira_target)
    # target_val,target_train=extract_loss2logit(lira_target)
    target_val= {key: value for key, value in target_val.items()}
    target_train= {key: value for key, value in target_train.items()}

    lira_train_idxs=list(range(50000))
    for idx in target_train.keys():
        lira_train_idxs.remove(idx)
    #print("len(lira_train_idxs)",len(lira_train_idxs))

    for k in lira_train_idxs:
        logit_train.append(train_dict[k])

    for k in range(50000):
        # if (k in target_train.keys())!= True:
        logit_val.append(val_dict[k])
        if k in target_train.keys():
            del logit_val[k][8]


    logit_train=np.array(logit_train)
    print(len(logit_val))
    logit_val=np.array(logit_val)

    mu_in=logit_train.mean(axis=1)
    mu_out=logit_val.mean(axis=1)

    var_in=logit_train.var() # global variance
    var_out=logit_val.var()

    lira_target=torch.load(f.format(K,epch))

    target_val,target_train=extract_fn(lira_target)
    # target_val,target_train=extract_loss2logit(lira_target)
    target_val= {key: value for key, value in target_val.items()}
    target_train= {key: value for key, value in target_train.items()}
    logits_new=[]
    for k in range(50000):
        if k in target_val.keys():
            logits_new.append(target_val[k][0])
        elif k in target_train.keys():
            logits_new.append(target_train[k][0])
        else:
            assert 0
    logits_new=np.array(logits_new)

    #print("new:",len(logits_new))
    #l_in,l_out=liratio(mu_in,mu_out,var_in,var_out,logits_new)
    l_out=liratio(mu_in,mu_out,var_in,var_out,logits_new)

    liratios=1-l_out
    val_liratios=np.array([liratios[i] for i in target_val.keys()])
    train_liratios=np.array([liratios[i] for i in target_train.keys()])
    auc,log_auc,tprs=plot_auc(f,torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
    accs.append(lira_target["test_acc"])

    print(f"tprs:{tprs}")
    # print(f"{lira_target['test_acc']:.4f}\t{tprs['0.1']}\t{tprs['0.01']}\t{auc:.4f}\t{log_auc:.4f}")

    #print("test_acc:",lira_target["test_acc"])
    return accs,tprs,auc,log_auc,liratios

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
        # print(val_liratios)

        val_liratios=-np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_cos']
        train_liratios=-np.array([ i.cpu().item() for i in train_liratios ])

        auc,log_auc,tprs=plot_auc("cos_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
        print("__"*10,"cos_attack")
        print(f"tprs:{tprs}",log_auc)
        # print("test_acc:",target_res[taret_idx])
        print("__"*10,)

    elif attack_mode =="grad diff":
        if MODE=="test":
            val_liratios=target_res['test_diffs']
        elif MODE=="val":
            val_liratios=target_res['val_diffs']
        val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=target_res['tarin_diffs']
        train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("diff_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
        print("__"*10,"diff_attack")
        print(f"tprs:{tprs}",log_auc)
        # print("test_acc:",target_res[taret_idx])
        print("__"*10,)
    elif attack_mode =="loss based":
        if MODE=="test":
            val_liratios=-ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
        elif MODE=="val":
            val_liratios=-ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] )
        # val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
        train_liratios=-ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
        # train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
        auc,log_auc,tprs=plot_auc("loss_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
        print("__"*10,"loss_attack")
        print(f"tprs:{tprs}",log_auc)
        # print("test_acc:",target_res[taret_idx])
        print("__"*10,)

    return accs,tprs,auc,log_auc,(val_liratios,train_liratios)


def fig_out(x_axis_data, MAX_K,defence,seed,log_path, d,avg_d=None,other_scores=None,accs=None): 
    colors={
        "cosine attack":"r",
        "grad diff":"g",
        "loss based":"b",
        "lira":"y",
        "log_lira":"k",
            }
    fig = plt.figure(figsize=(6.5, 6.5), dpi=200)
    fig.subplots_adjust(top=0.91,
                        bottom=0.160,
                        left=0.180,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    for k in d.keys():
        plt.plot(x_axis_data[0:len(d[k])], d[k], linewidth=1, label=k)#,color=colors[k])
    # plt.plot(x_axis_data, common_score,'bo-', linewidth=1, color='#2E8B57', label=r'Baseline')
    plt.legend(loc=3)  

    plt.xlim(-2, 305)
    my_x_ticks = np.arange(0, 302, 50)
    plt.xticks(my_x_ticks,size=14)
    if avg_d:
        for k in avg_d.keys():
            if avg_d[k]:    
                plt.hlines([avg_d[k]["0.01"]],xmin=0,xmax=300,label=k,color=colors[k])

    plt.legend(prop={'size': 10})
    plt.xlabel('Epoch',fontsize=14,fontdict={'size': 14})  # x_label
    plt.ylabel('TPR@FPR=0.01',fontsize=14,fontdict={'size': 14})  # y_label
    plt.grid(axis='both')

    pdf_path=PATH.split("/")[0:-1]
    pdf_path="/".join(pdf_path)+"/attack.pdf"
    plt.savefig(pdf_path)

    print("log_path0:",log_path)
    log_path=log_path+f"/def{defence}2_0.85_k{MAX_K}_{seed}_attack.log"
    print("log_path:",log_path)
    with open(log_path,"w") as f:
        json.dump({"avg_d":avg_d,"other_scores":other_scores,"accs":accs},f)
    # assert 0

@ torch.no_grad()
def attack_comparison(p,log_path,epochs,MAX_K,defence,seed):
    
    final_acc=lira_attack_ldh_cosine(p,epochs[-1],K=MAX_K,extract_fn=extract_hinge_loss)[0]
    reses_lira=[]
    reses_common={k:[] for k in attack_modes}
    lira_scores=[]
    common_scores=[]
    cos_scores=[]
    other_scores={}
    scores={k:[] for k in attack_modes}
    scores["lira"]=[]
    avg_scores={k:None for k in attack_modes}
    avg_scores["lira"]=None
    for epch in epochs:
        print("test0:",epch)
        try:
            lira_score=lira_attack_ldh_cosine(p,epch,K=MAX_K,extract_fn=extract_hinge_loss)
        except ValueError:
            print("ValueError")
            continue
        scores["lira"].append(lira_score[1]['0.01'])
        # lira_score=lira_attack(p,epch,K=9,extract_fn=extract_hinge_loss)
        for attack_mode in attack_modes:
            common_score=cos_attack(p,0,epch,attack_mode,extract_fn=extract_hinge_loss)
            reses_common[attack_mode].append(common_score[-1])
            scores[attack_mode].append(common_score[1]['0.01'])
            if epch ==200 and attack_mode=="loss based":
                other_scores["loss_single_epch_score"]=common_score[1]
                other_scores["loss_single_auc"]=[common_score[2],common_score[3]]


        lira_scores.append(lira_score[1]['0.01'])
        common_scores.append(common_score[1]['0.01'])

        reses_lira.append(lira_score[-1])

    reses=reses_lira
    train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    #print(train_score.shape)
    # assert 0
    auc,log_auc,tprs=plot_auc("averaged_lira",torch.tensor(test_score),torch.tensor(train_score),999)
    print(f"tprs:{tprs}")
    avg_scores["lira"]=tprs
    other_scores["lira_auc"]=[auc,log_auc]
    print("success!")

    reses=reses_lira
    train_score=-np.nanmean(1-np.log(np.vstack([ i[0].reshape(1,-1) for i in reses])),axis=0)
    test_score=-np.nanmean(1-np.log(np.vstack([ i[1].reshape(1,-1) for i in reses])),axis=0)
    print(train_score.min(),train_score.max())
    train_score=train_score[~(np.isnan(train_score))]
    test_score=test_score[~(np.isnan(test_score))]
    train_score[(np.isinf(train_score))]=-1e10
    test_score[(np.isinf(test_score))]=-1e10
    print(train_score.min(),train_score.max())
    print(train_score.shape)
    # assert 0
    auc,log_auc,tprs=plot_auc("averaged_log_lira",torch.tensor(test_score),torch.tensor(train_score),999)
    print(f"tprs:{tprs}")
    avg_scores["log_lira"]=tprs
    other_scores["log_lira_auc"]=[auc,log_auc]
    print("success!")



    reses=reses_common["cosine attack"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=-np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=-np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    auc,log_auc,tprs=plot_auc("cosine attack",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["cosine attack"]=tprs
    other_scores["cos_attack_auc"]=[auc,log_auc]
    print(f"tprs:{tprs}")
    print("success!")

    reses=reses_common["grad diff"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=-np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=-np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    auc,log_auc,tprs=plot_auc("averaged_diff",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["grad diff"]=tprs
    other_scores["grad_diff_auc"]=[auc,log_auc]
    print(f"tprs:{tprs}")
    print("success!")

    reses=reses_common["loss based"]
    # print(reses)
    # print(len(reses),len(reses[0]))
    # assert 0
    train_score=-np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
    test_score=-np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
    auc,log_auc,tprs=plot_auc("averaged_loss",torch.tensor(test_score),torch.tensor(train_score),999)
    avg_scores["loss based"]=tprs
    other_scores["loss_based_auc"]=[auc,log_auc]
    print(f"tprs:{tprs}")
    print("success!")

    fig_out(epochs,MAX_K, defence,seed,log_path,scores,avg_scores,other_scores,final_acc)

def main(argv):

    global MODE, attack_modes, PATH,p_folder
    MODE='val' #"val"
    attack_modes=["cosine attack","grad diff","loss based"]
    epochs=list(range(10,int(argv[2])+1,10))
    p_folder=argv[1]  
    PATH=argv[1]
    global device
    device=argv[3] 

    # log_path="logs/log_res"
    MAX_K=10

    for root, dirs, files in os.walk(p_folder, topdown=False):
        for name in dirs:
            #print("names:",name)
            if len(name.split("_"))<7 or len(name.split("_")[-1])>5 :
                #print("【Error】:",os.path.join(root, name))
                continue
            elif root==p_folder:
                print(os.path.join(root, name))
                PATH=os.path.join(root, name)
                PATH+="/client_{}_losses_epoch{}.pkl"
                MAX_K=int(name.split("_K")[1].split("_")[0])
                model=name.split("_")[3]
                defence=name.split("_")[-5].strip('def').strip('0.0')
                seed=name.split("_")[-1]
                print("name:",name)
                if model== "alexnet":
                    log_path="logs/log_alex"
                else:
                    log_path="logs/log_res"
                # print(MAX_K,PATH)
                try:
                    attack_comparison(PATH,log_path,epochs, MAX_K,defence,seed)
                    print("success!")
                except IOError:
                    print("error:",MAX_K,PATH)
                    pass
            print(os.path.join(root, name))

if __name__ == "__main__":
    main(sys.argv)
