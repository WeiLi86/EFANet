CUDA_VISIBLE_DEVICES=0 python -u -m code.valid \
                       --model_type EFANet \
                       --data_path /home/liwei/project/data --data avazu \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path /home/liwei/project/model/EFANet/avazu/ \
                       --field_size 23  --run_times 1 \
                       --epoch 4 --batch_size 2048 \

