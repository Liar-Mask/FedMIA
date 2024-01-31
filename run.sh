#!/bin/bash

#TB1 
# python main_orign.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --gpu 0 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'milestone'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/0828_stone_SGD/10_clients/'

#
# python main_orign.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --gpu 1 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/0830_niid_cosine_SGD_lr0.01/10_clients/beta0.01/'


#TB2
# python main_orign.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --gpu 3 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/0830_niid_fix_adam/10_clients/beta0.1/'

#TB3
# python main_orign.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name resnet\
#  --epochs 300 --gpu 2 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'cosine' \
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/0830_niid_coslr_SGD_res_lr0.01/10_clients/beta0.01/'

#TB4
# python main_orign.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name resnet\
#  --epochs 300 --gpu 0 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/0830_niid_fix_adam_res/10_clients/beta0.01/'


# Correct Alex iid
# CUDA_VISIBLE_DEVICES=0 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.2 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/correct99/iid/alex/'\
#  --log_folder_name '/training_log_correct_iid/'

# Correct Alex non-iid
# CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
<<<<<<< HEAD
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/correct99/non-iid/alex/'\
=======
#  --save_dir '../tmp'\
>>>>>>> b84b2379f7c303459f21d7217811344902c7015e
#  --log_folder_name '/training_log_correct_noniid/'

# Correct Resnet iid
#  CUDA_VISIBLE_DEVICES=2 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/correct99/iid/res/'\
#  --log_folder_name '/training_log_correct_iid/'

# Correct Resnet Noniid

#  CUDA_VISIBLE_DEVICES=3 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/correct99/noniid/'\
#  --log_folder_name '/training_log_correct_noniid/'


# Correct Alex non-iid
# CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '../tmp'\
#  --log_folder_name '/training_log_correct_noniid/'


# Correct Resnet iid

 CUDA_VISIBLE_DEVICES=0 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine' \
 --save_dir '../tmp_resnet_iid_lr01' --iid \
 --log_folder_name '/training_log_correct_iid/' &

  CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp_resnet_iid_adam' --iid \
 --log_folder_name '/training_log_correct_iid/' &

 CUDA_VISIBLE_DEVICES=2 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
 --save_dir '../tmp_alexnet_iid_lr01' --iid \
 --log_folder_name '/training_log_correct_iid/' &

  CUDA_VISIBLE_DEVICES=3 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp_alexnet_iid_adam' --iid \
 --log_folder_name '/training_log_correct_iid/' &
 


#  CUDA_VISIBLE_DEVICES=0 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine' \
#  --save_dir '../tmp_resnet_iid_lr01' --iid \
#  --log_folder_name '/training_log_correct_iid/' &

#   CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_resnet_iid_adam' --iid \
#  --log_folder_name '/training_log_correct_iid/' &

#  CUDA_VISIBLE_DEVICES=2 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '../tmp_alexnet_iid_lr01' --iid \
#  --log_folder_name '/training_log_correct_iid/' &

#   CUDA_VISIBLE_DEVICES=3 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_alexnet_iid_adam' --iid \
#  --log_folder_name '/training_log_correct_iid/' &
 

# ldh version 
  CUDA_VISIBLE_DEVICES=9 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp2_alexnet_iid_adam' --iid \
 --log_folder_name '/training_log_correct_iid/' &

   CUDA_VISIBLE_DEVICES=8 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp2_resnet18_iid_adam' --iid \
 --log_folder_name '/training_log_correct_iid/' &

   CUDA_VISIBLE_DEVICES=7 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp2_alexnet_iid_sgd' --iid \
 --log_folder_name '/training_log_correct_iid/' &

   CUDA_VISIBLE_DEVICES=6 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp2_resnet18_iid_sgd' --iid \
 --log_folder_name '/training_log_correct_iid/' &



#    CUDA_VISIBLE_DEVICES=0 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_alexnet_iid_adam' --iid --data_augment \
#  --log_folder_name '/training_log_correct_iid_aug/' &

#    CUDA_VISIBLE_DEVICES=1 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_resnet18_iid_adam' --iid --data_augment \
#  --log_folder_name '/training_log_correct_iid_aug/' &

#    CUDA_VISIBLE_DEVICES=2 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_alexnet_iid_sgd' --iid --data_augment \
#  --log_folder_name '/training_log_correct_iid_aug/' &

#    CUDA_VISIBLE_DEVICES=4 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_resnet18_iid_sgd' --iid --data_augment \
#  --log_folder_name '/training_log_correct_iid_aug/' &


wait


CUDA_VISIBLE_DEVICES=1 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset dermnet --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/dermnet_resnet18_iid_sgd' --iid \
 --log_folder_name '/dermnet_adam_iid/' &

 CUDA_VISIBLE_DEVICES=2 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset dermnet --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/dermnet_alexnet_iid_sgd' --iid \
 --log_folder_name '/dermnet_adam_iid/' &

 CUDA_VISIBLE_DEVICES=3 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset oct --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/oct_resnet18_iid_sgd' --iid \
 --log_folder_name '/oct_adam_iid/' &

 CUDA_VISIBLE_DEVICES=4 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset oct --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/oct_alexnet_iid_sgd' --iid \
 --log_folder_name '/oct_adam_iid/' &



 CUDA_VISIBLE_DEVICES=9 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset dermnet --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/dermnet_resnet18_iid_sgd_small' --iid \
 --log_folder_name '/dermnet_adam_iid/' &

 CUDA_VISIBLE_DEVICES=8 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset dermnet --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/dermnet_alexnet_iid_sgd_small' --iid \
 --log_folder_name '/dermnet_adam_iid/' &

 CUDA_VISIBLE_DEVICES=7 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset oct --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/oct_resnet18_iid_sgd_small' --iid \
 --log_folder_name '/oct_adam_iid/' &

 CUDA_VISIBLE_DEVICES=6 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset oct --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../FLMIA_res/oct_alexnet_iid_sgd_small' --iid \
 --log_folder_name '/oct_adam_iid/' &


dataset=cifar100
model_name=alexnet
opt=adam
local_epoch=1
save_dir=../FLMIA_res_0912/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0912/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

dataset=dermnet
model_name=alexnet
local_epoch=1
save_dir=../FLMIA_res_0912/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0912/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &


dataset=cifar100
model_name=alexnet
opt=adam
lr=0.001
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=9 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=8 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

dataset=dermnet
model_name=alexnet
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=7 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=6 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &


dataset=oct
model_name=alexnet
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=4 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=5 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &


dataset=oct
model_name=alexnet
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=4 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=5 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &


dataset=cifar100
model_name=alexnet
opt=sgd
lr=0.1
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

dataset=dermnet
model_name=alexnet
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

dataset=oct
model_name=alexnet
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=6 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=7 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &



# 0913
dataset=cifar100
model_name=ResNet18
opt=sgd
lr=0.1
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=6 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

dataset=dermnet
model_name=ResNet18
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

dataset=oct
model_name=ResNet18
local_epoch=1
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=4 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &
local_epoch=2
save_dir=../FLMIA_res_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
CUDA_VISIBLE_DEVICES=5 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr 0.001 --batch_size 100 --optim $opt --lr_up 'cosine' --save_dir $save_dir --iid  --log_folder_name $save_dir &

