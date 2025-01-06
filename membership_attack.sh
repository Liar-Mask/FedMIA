# membership inference attack command
path="log_fedmia/iid"
seed=2025
total_epoch=300
gpu=1
python -u mia_attack_auto.py  ${path} ${total_epoch} ${gpu} ${seed}  

