
#not resume
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12355 train.py --batch-size 16 --epochs 100 --lr 1e-5 --save_interval 30
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12355 train.py --batch-size 16 --epochs 100 --lr 1e-4 --save_interval 30



CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12355 train.py --batch-size 16 --epochs 100 --lr 1e-4 --save_interval 30


#resume
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12355 train.py --batch-size 16 --epochs 100 --lr 1e-5 --save_interval 30 --RESUME True --ckpt_file 
