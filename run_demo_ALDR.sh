
CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.0 | tee patch/letter_clean_ALDR_mom_1e-3_decay=5e-3

CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.5 | tee patch/letter_class-dependent=0.5_ALDR_mom_1e-3_decay=5e-3

CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.3 | tee patch/letter_class-dependent=0.3_ALDR_mom_1e-3_decay=5e-3

CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=class-dependent --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.1 | tee patch/letter_class-dependent=0.1_ALDR_mom_1e-3_decay=5e-3

CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.9 | tee patch/letter_uniform=0.9_ALDR_mom_1e-3_decay=5e-3

CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.6 | tee patch/letter_uniform=0.6_ALDR_mom_1e-3_decay=5e-3

CUDA_VISIBLE_DEVICES=0  python3 experiment.py --dataset=letter --loss=ALDR  --noise_type=uniform --lr=1e-3  --decay=5e-3 --alpha=2 --noise_level=0.3 | tee patch/letter_uniform=0.3_ALDR_mom_1e-3_decay=5e-3


