# LDR-KL and ALDR-KL loss functions
The LDR-KL loss and ALDR-KL loss from: 'Label Distributionally Robust Losses for Multi-class Classification: Consistency, Robustness and Adaptivity', Dixian Zhu, Yiming Ying, Tianbao Yang, International Conference on Machine Learning (ICML), 2023.

This is the code that runs LDR-KL and ALDR-KL loss on benchmark datasets, such as Vowel, Letter, Kuzushiji-49, CIFAR-100, etc. 

## Examples: 

### Run LDR-KL and ALDR-KL losses on clean Letter data

```
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=LDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --noise_level=0.0
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.0
```

### Run LDR-KL and ALDR-KL losses on Letter data with uniform noise as 0.6

```
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=LDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --noise_level=0.6
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.6
```

### Run LDR-KL and ALDR-KL losses on Letter data with class dependent noise as 0.3

```
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=LDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --noise_level=0.3
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.3
```

## Others:
Please make sure you have the data on the data folder. Please refer to the experiment.py for how to load the data. 

For synthetic data experiments, please refer to the original paper for a simple setup. For the real world noisy dataset, mini-webvision, we thanks the previous pipeline [Ma et al](https://github.com/HanxunH/Active-Passive-Losses) and [Zhou et al](https://github.com/hitcszx/ALFs) where we further implement our loss functions. 

## Citation:
```
@inproceedings{zhu2023label,
	title={Label Distributionally Robust Losses for Multi-class Classification: Consistency, Robustness and Adaptivity},
	author={Zhu, Dixian and Ying, Yiming and Yang, Tianbao},
	booktitle={Proceedings of the 40th International Conference on Machine Learning},
	year={2023}
	}  
```
