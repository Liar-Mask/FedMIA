# federated training and attack measurement calculatinging command
dataset=cifar100 # or dermnet
model_name=alexnet # or ResNet18

opt=sgd
seed=1
lr=0.1
local_epoch=1

# iid experiment
save_dir=log_fedmia/iid
mkdir -p "$save_dir"
CUDA_VISIBLE_DEVICES=1 python main.py --seed $seed --num_users 10 --iid 1   \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep $local_epoch \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --lr_up cosine --MIA_mode 1  --gpu 1 &> "${save_dir}/raw_logs" &

# non-iid experiment
save_dir=log_fedmia/noniid
mkdir -p "$save_dir"
CUDA_VISIBLE_DEVICES=0 python main.py --seed $seed --num_users 10 --iid 0 --beta 1.0  \
 --dataset $dataset --model_name $model_name --epochs 300 --local_ep $local_epoch \
 --lr $lr --batch_size 100 --optim $opt --save_dir $save_dir --log_folder_name $save_dir \
 --lr_up cosine --MIA_mode 1  --gpu 0 2>&1 | tee "${save_dir}/raw_logs" &

wait
