CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun --nproc-per-node=4 \
--master_port=9742 \
-m examples.resnet.train \
--dataset_download_path=./examples/resnet/datasets/ \
--model_type=resnet18 \
--epochs=2 \
--learning_rate=1e-4 \
--total_train_batch_size=32 \
--total_dev_batch_size=4 
