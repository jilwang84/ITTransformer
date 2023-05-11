__COMP5214 Course Project: Inter-sample Tabular Transformer__

This project unified the implementation of TabTransformer, FTTransformer, and the proposed ITTransformer. By default, six classical tabular datasets from OpenML and AutoML can be selected to conduct training. 

**Usage**

Training and Testing:

`python main.py --model %s --batch %s --dataset %s --epoch %s --dropout %s --lr %s --optimizer %s --scheduler %s --early_stop %s --loss %s`

- `model` Available model: TabTransformer, FTTransformer, ITTransformer. Default: ITTransformer.
- `batch` Batch size. Default 256.
- `dataset` Dataset name: adult, covertype, income, bank, volkert, Diabetes130US. Default adult.
- `epoch` The number of epochs for training. Default 100.
- `dropout` Dropout rate. Default 0.0.
- `lr` Learning rate. Default 0.0001.
- `optimizer` Use optimizer for training: AdamW, Adam, RAdam. Default AdamW.
- `scheduler` Use scheduler for training: cosine, linear, cosine_warmup, none. Default none.
- `early_stop` Early stopping patience. Default None.
- `loss` Loss function. Default CrossEntropy.

There are more arguments that can be adjusted, please read the argument parsing section of `main.py` for more details. 

Training and testing logs and results will be stored in the folder `log` and `result` respectively.

Required packages:
```
argparse
datetime
einops
logging
matplotlib
numpy
os
pandas
time
torch
tqdm
```

**Citation**

```
@article{huang2020tabtransformer,
  title={Tabtransformer: Tabular data modeling using contextual embeddings},
  author={Huang, Xin and Khetan, Ashish and Cvitkovic, Milan and Karnin, Zohar},
  journal={arXiv preprint arXiv:2012.06678},
  year={2020}
}
@article{gorishniy2021revisiting,
  title={Revisiting deep learning models for tabular data},
  author={Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={18932--18943},
  year={2021}
}
```

