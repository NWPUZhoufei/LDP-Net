# LDP-net
Code and model for "Revisiting Prototypical Network for Cross Domain Few-Shot Learning" (LDP-net).

# Requirements
- numpy==1.16.4
- scipy==1.3.0
- scikit-learn==0.21.2
- torch==1.8.0+cu111
- torchaudio==0.8.0
- torchvision==0.9.0+cu111
- python 3.7.3

# Datesets
Following "A Broader Study of Cross-Domain Few-Shot Learning" - > EuroSAT, CropDisease, ISIC, Chest.
Following "CROSS-DOMAIN FEW-SHOT CLASSIFICATION VIA LEARNED FEATURE-WISE TRANSFORMATION"  - > CUB, cars, Places, Plantae.
Please refer to the above works to obtain datasets or you can download these datasets from this (address: https://pan.baidu.com/s/1xOEQuT1jP6Z1QIVZkMTHng,
password: ao3m) , which I processed the datasets as described in these papers.

# Train
```
CUDA_VISIBLE_DEVICES=0  nohup python train.py --lamba1 1.0 --lamba2 0.15 --m 0.998 --seed 1111 --epoch 100 --train_n_eposide 100 --n_support 5 --source_data_path ./source_domain/miniImageNet/train  --pretrain_model_path  ./pretrain/399.tar  --save_dir checkpoint >record.log 2>&1 &

```

# Test for EuroSAT:
```
- 5-way 1-shot:
CUDA_VISIBLE_DEVICES=0  nohup python test.py --n_support 1 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600  --model_path ./checkpoint/100.tar  >record_t1.log 2>&1 &

- 5-way 5-shot:
CUDA_VISIBLE_DEVICES=0  nohup python test.py --n_support 5 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600  --model_path ./checkpoint/100.tar  >record_t2.log 2>&1 &

- Test exploiting the full data of few-shot task:
- 5-way 1-shot:
CUDA_VISIBLE_DEVICES=0  nohup python test_tr_1shot.py --n_support 1 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600   --model_path ./checkpoint/100.tar  >record_t3.log 2>&1 &

- 5-way 5-shot:
CUDA_VISIBLE_DEVICES=0  nohup python test_tr_5shot.py --n_support 5 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600   --model_path ./checkpoint/100.tar  >record_t4.log 2>&1 &

```
# Test for others:
```
-Test for CropDisease：
replace：--current_data_path ./target_domain/CropDisease  --current_class 38  

- Test for ISIC：
replace：--current_data_path ./target_domain/ISIC --current_class 7

- Test for CUB：
replace：--current_data_path ./target_domain/CUB/novel  --current_class 50

- Test for cars：
replace：--current_data_path ./target_domain/cars/novel  --current_class 49

- Test for Places：
replace：--current_data_path ./target_domain/Places/novel  --current_class 91

- Test for Plantae：
replace：--current_data_path ./target_domain/Plantae/novel  --current_class 50

- Test for Chest：
replace：--current_data_path ./target_domain/Chest7  --current_class 7

```
# Thanks
Thanks to the authors of these works (https://github.com/IBM/cdfsl-benchmark and https://github.com/hytseng0509/CrossDomainFewShot) for providing the code framework. 

# Any questions:
If you have any questions, please leave a message in github-issues, or send an email to zhoufei@mail.nwpu.edu.cn

