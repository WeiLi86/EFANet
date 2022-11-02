CUDA_VISIBLE_DEVICES=1 python -u -m code.train \
                       --model_type EFANet \
                       --data_path /home/liwei/project/data --data criteo \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path /home/liwei/project/save/EFANet/criteo/0/ \
                       --field_size 39  --run_times 2 \
                       --epoch 2 --batch_size 2048 \

