# OFF-NET original
python3 train.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --batch_size 12  --gpu_ids 0,1,2,3,4,5
python3 test.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --prob_map  --epoch best
python3 test.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --prob_map  --epoch last
# Clip Pretrained Swin-T
python3 train.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --batch_size 12  --gpu_ids 0,1,2,3,4,5 --extra_v_encoder --pretrain_model_path '/home/yangyi/wd/paper/checkpoint/Clip_240_9.cpt' --fix_v_encoder
python3 test.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --prob_map  --epoch best --extra_v_encoder --pretrain_model_path '/home/yangyi/wd/paper/checkpoint/Clip_240_9.cpt' --fix_v_encoder
python3 test.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --prob_map  --epoch last --extra_v_encoder --pretrain_model_path '/home/yangyi/wd/paper/checkpoint/Clip_240_9.cpt' --fix_v_encoder
# only Clip Pretrained Swin-T as encoder
python3 train.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --batch_size 12  --gpu_ids 0,1,2,3,4,5 --extra_v_encoder --pretrain_model_path '/home/yangyi/wd/paper/checkpoint/Clip_240_9.cpt' --fix_v_encoder --only_extra_v_encoder
python3 test.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --prob_map  --epoch best --extra_v_encoder --pretrain_model_path '/home/yangyi/wd/paper/checkpoint/Clip_240_9.cpt' --fix_v_encoder --only_extra_v_encoder
python3 test.py --dataroot '/mnt/filesystem4/yangyi/ORFD' --dataset ORFD --name ORFD --use_sne --prob_map  --epoch last --extra_v_encoder --pretrain_model_path '/home/yangyi/wd/paper/checkpoint/Clip_240_9.cpt' --fix_v_encoder --only_extra_v_encoder
