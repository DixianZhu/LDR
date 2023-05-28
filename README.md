# LDR-KL and ALDR-KL loss functions
The LDR-KL loss and ALDR-KL loss from: 'Label Distributionally Robust Losses for Multi-class Classification: Consistency, Robustness and Adaptivity', Dixian Zhu, Yiming Ying, Tianbao Yang, International Conference on Machine Learning (ICML), 2023.

This is the code that how to run LDR-KL and ALDR-KL loss on benchmark datasets, such as Vowel, Letter, Kuzushiji-49, CIFAR-100, etc. 

## Examples: 

### Run LDR-KL and ALDR-KL losses on clean Letter data

- CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=LDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --noise_level=0.0
- CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.0

### Run LDR-KL and ALDR-KL losses on Letter data with uniform noise as 0.6

- CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=LDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --noise_level=0.6
- CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.6

### Run LDR-KL and ALDR-KL losses on Letter data with class dependent noise as 0.3

- CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=LDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --noise_level=0.3
- CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.3
