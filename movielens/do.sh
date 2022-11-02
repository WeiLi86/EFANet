CUDA_VISIBLE_DEVICES=2 python -u -m movie_code.train \
                       --model_type FRANet \
                       --data_path /home/liwei/project/data/movie  \
                       --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save --has_residual \
                       --save_path /home/liwei/project/save/FRANet/movie/2/ \
                       --field_size 7  --run_times 5 \
                       --dropout_keep_prob "[0.6, 0.9]" \
                       --epoch 40 --batch_size 512 \
