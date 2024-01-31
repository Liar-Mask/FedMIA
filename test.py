import numpy as np
import os
from utils.args import parser_args
# from utils.help import *
from utils.datasets import *
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# MIA_valset_dir=[]

# MIA_trainset = [0,1,2]

# MIA_valset = [4,5,6]

# MIA_trainset_dir.append(MIA_trainset)
# MIA_trainset_dir.append(MIA_valset)

# print(MIA_trainset_dir[1])


# for epch in range(10,301,10):
#     p=f"/CIS32/zgx/Fed2/Code/MIA_Log/0825_test2/10_clients/client_{}_losses_epoch{epch}.pkl"
#     print(p)

# x = torch.tensor([1., 2.], requires_grad=True)
# # x: tensor([1., 2.], requires_grad=True) 
# y = 100*x
# # y: tensor([100., 200.], grad_fn=<MulBackward0>)

# loss = y.sum() # tensor(300., grad_fn=<SumBackward0>)

# # Compute gradients of the parameters respect to the loss
# #print(x.grad)     # None, 反向传播前，梯度不存在
# loss.backward()      
# print("x{}\n y{}".format(x.grad,10*x.grad))     # tensor([100., 100.]) loss对y的梯度为1， 对x的梯度为100
# print(x.grad.data)


# optim = torch.optim.SGD([x], lr=0.001) # 随机梯度下降， 学习率0.001
# print(x)        # tensor([1., 2.], requires_grad=True)
# optim.step()  # 更新x
# print(x)        # tensor([0.9000, 1.9000], requires_grad=True) 变化量=梯度X学习率 0.1=100*0.001


# a=torch.rand(4, 3, 3, 11).flatten(start_dim=1)
# b=torch.rand(4,2,2).flatten(start_dim=1)
# c=torch.cat((a,b),1)
# #print(c.shape)

# d=torch.rand(4,12)
# e=torch.rand(4,12)
# f=torch.cat((d,e),0)
# print(f.shape)
# for i in range(4):
#     cos=  F.cosine_similarity(a[i] ,b, dim=0)
#     print(cos)
# PATH="/CIS32/zgx/Fed2/Code/MIA_Log/ICLR2023_0913/cifar100_ResNet18_iid_adam_local1/client_{}_losses_epoch{}.pkl"
# pdf_path=PATH.split("/")[0:-1]
# pdf_path="/".join(pdf_path)+"/attack.pdf"
# print(pdf_path)

print(os.path.join(os.getcwd(), 'MIA_test/log','1.pkl'))
print(os.path.join(os.getcwd(), 'MIA_test/log','pkl_files/', 'client_{}_losses_epoch{epoch+1}.pkl'))



def mixup_data(self, x, y,alpha):
        
    use_cuda=True
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
alpha=1e-6
print( np.random.beta(alpha, alpha))