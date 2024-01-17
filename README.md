#Privacy-Preserving Proof-of-Learning via Watermark Trajectory

## How to run
Python run.py 

Train GTSRB dataset
```
python main.py --dataset GTSRB --num_class 43 --a 0.3 --b 0.1 --weight_decay 0
```
Train CelebA dataset
```
python main.py --dataset CelebA --num_class 8 --a 0.3 --b 0.1 --weight_decay 1e-4
```

## Tips
Download data from Baidudisk(code:sft2)/Google driver and unzip it to root folder.  
https://drive.google.com/file/d/1o_T6VNS8FHu1EDvBKEjw92agbK1sD0p9/view?usp=sharing
