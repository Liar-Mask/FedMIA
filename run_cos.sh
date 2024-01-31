
# run cosine attack--Alex
# CUDA_VISIBLE_DEVICES=2 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'common'\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/ICLR2023_0911/alex_sgd_0.01_common/only1/testldr/'\
#  --log_folder_name '/training_log/cos_attck0912/'

## run cosine attack--Alex noniid
# CUDA_VISIBLE_DEVICES=1 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'common' --beta 1\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/ICLR2023_0911/alex_sgd_0.01_common/non-iid/'\
#  --log_folder_name '/training_log/cos_attck0911/non-iid/'

# run cosine attack--Res

# CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.01 --batch_size 100 --optim 'sgd' --lr_up 'common' --beta 0.1\
#  --save_dir '/CIS32/zgx/Fed2/Code/MIA_Log/ICLR2023cos/res_sgd_0.01_common/'\
#  --log_folder_name '/training_log/cos_attck0909/'


dataset=cifar100
model_name=ResNet18
opt=adam
lr=0.001
# local_epoch=1
# save_dir=../MIA_Log/ICLR2023_0913/${dataset}_${model_name}_iid_${opt}_local${local_epoch}
# CUDA_VISIBLE_DEVICES=0 python main_ldh.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 \
#  --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt\
#   --lr_up 'cosine' --save_dir $save_dir --log_folder_name $save_dir 


# dataset=cifar100
# model_name=alexnet
# local_epoch=1
# lr=0.1
# save_dir=/MIA_Log/mix_up_1015/
# #${dataset}_${model_name}_iid_${opt}_local${local_epoch}
# CUDA_VISIBLE_DEVICES=1 python main_ldh.py  --seed $seed --num_users 10  --mix_up \
#  --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr  $lr --batch_size 100 --optim sgd\
#   --lr_up 'cosine' --save_dir $save_dir --log_folder_name $save_dir &

# CUDA_VISIBLE_DEVICES=2 python main_ldh.py  --seed $seed --num_users 10  \
#  --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr  $lr --batch_size 100 --optim sgd\
#   --lr_up 'cosine' --save_dir $save_dir --log_folder_name $save_dir  &

## instahide
dataset=cifar100
model_name=ResNet18
opt=sgd
seed=1 #lambda.max > 0.5
lr=0.1
local_epoch=1

defense=instahide


klam=3
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up common --defense $defense --MIA_mode 1 &

klam=3
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up common --defense $defense --MIA_mode 1 &

klam=2
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up common --defense $defense --MIA_mode 1 &

klam=2
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up common --defense $defense --MIA_mode 1 &




dataset=cifar10
model_name=alexnet #ResNet18
opt=sgd
seed=11 #lambda.max > 0.5
lr=0.01
klam=3
local_epoch=2
save_dir=/log_defense/instahide$klam
defense=instahide
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt\
 --klam $klam --lr_up common --save_dir $save_dir --log_folder_name $save_dir --defense $defense



dataset=cifar100
model_name=ResNet18
opt=sgd
seed=11 #lambda.max >0.5
lr=0.1
local_epoch=1
save_dir=log_defense/mix_up
defense=mix_up
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} --lr $lr --batch_size 100 --optim $opt\
 --lr_up common --save_dir $save_dir --log_folder_name $save_dir --defense $defense



#MIA1 --klam=3 Resnet CIFAR10
#MIA2 --klam=2 Resnet CIFAR10
#MIA3 --mix_up alexnet CIFAR10
#MIA4 --mix_up alexnet CIFAR100