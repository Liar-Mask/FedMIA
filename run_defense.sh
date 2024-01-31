
## instahide
dataset=cifar100
model_name=ResNet18
opt=sgd
seed=1 #lambda.max > 0.5
lr=0.1
local_epoch=1

defense=instahide

# klam in {2, 3, 4}
# up_bound in {0.65, 0.85}
klam=3 
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=3
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=2
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=2
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=4
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=4 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=4
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=5 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &




dataset=cifar100
model_name=alexnet #ResNet18
opt=sgd
seed=1 #lambda.max > 0.5
lr=0.1
defense=instahide

#klam in {2, 3, 4}
#up_bound in {0.65, 0.85}
klam=3 
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=3
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=2
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=2
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=4
up_bound=0.65
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=4 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &

klam=4
up_bound=0.85
save_dir=log_defense/instahide${klam}_up${up_bound}
CUDA_VISIBLE_DEVICES=5 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --klam $klam --up_bound $up_bound --lr_up cosine --defense $defense --MIA_mode 1 &


# none defense test

dataset=cifar100
model_name=ResNet18
opt=sgd
seed=5 #lambda.max >0.5
lr=0.1
local_epoch=1
defense=none
save_dir=log_defense/none
CUDA_VISIBLE_DEVICES=3 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --lr_up cosine --defense $defense --MIA_mode 1 

## mix_up defense
dataset=cifar100
model_name=alexnet  #ResNet18
opt=sgd
seed=5 #lambda.max >0.5
lr=0.1
local_epoch=1
defense=mix_up

# /CIS32/zgx/Fed2/Code/FedMIA2/log_defense/mix_alpha_1
#mix_alpha in {1e-7,1e-5,1e-3,1e-2,1e-1,1,5,10,20,50,100}
alpha=1
save_dir=log_defense_0113/mix_alpha_${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=0.000001
save_dir=log_defense_0113/mix_alpha_${alpha}
CUDA_VISIBLE_DEVICES=1 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=1000000
save_dir=log_defense/mix_alpha_${alpha}
CUDA_VISIBLE_DEVICES=2 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
 
alpha=0.01
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=0.1
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=1
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=5
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=10
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=20
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=50
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=100
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &


dataset=cifar100
model_name=alexnet
opt=sgd
seed=1 #lambda.max >0.5
lr=0.1
local_epoch=1
defense=mix_up

#mix_alpha in {1e-7,1e-5,1e-3,1e-2,1e-1,1,5,10,20,50,100}
alpha=0.0000001
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=0.00001
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=0.001
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
 
alpha=0.01
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=0.1
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &

alpha=1
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=5
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=10
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=20
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=50
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &
alpha=100
save_dir=log_defense/mix_alpha${alpha}
CUDA_VISIBLE_DEVICES=0 python main_ldh.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep ${local_epoch} \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --mix_alpha $alpha --lr_up cosine --defense $defense --MIA_mode 1 &