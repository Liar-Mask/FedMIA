
dataset=cifar100
model_name=alexnet
opt=sgd
seed=1 
lr=0.1
local_epoch=1

save_dir=log_fedmia
CUDA_VISIBLE_DEVICES=0 python main.py --seed $seed --num_users 10 --iid 1 \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep $local_epoch \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --lr_up cosine --MIA_mode 1 