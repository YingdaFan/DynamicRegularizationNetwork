# Dynamic Regularization 
Dynamic Regularization Network for Transformer in Neural Machine Translation 
Introduced a sparsity regulation term bounded with multi-head attention output.
The normalized vector in the regularization term rectified and regularized the final loss function.
Regularization changed dynamically with training, which depends on the dynamic normalized vector.
Visual analysis of attention weight showed that dynamic regularization sharpened the attention weight distribution.

### Requirements
* [PyTorch](http://pytorch.org/) version == 1.9
* Python version >= 3.8.8   
* [Fairseq](https://github.com/facebookresearch/fairseq/) version == 0.10.1
### Train
Our method is based on [fairseq toolkit](https://github.com/pytorch/fairseq) for training and evaluating. 
The bilingual datasets [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf) and [WMT'14 English to German dataset](http://www.statmt.org/wmt14/translation-task.html) should be first preprocessed into binary format and save in 'data-bin' file. 

```
cd ./fairseq
user_dir=./regularization
data_bin=./data-bin/wmt14_en_de_bpe32k
model_dir=./models/regularization
export CUDA_VISIBLE_DEVICES=1
nohup fairseq-train $data_bin \
        --user-dir $user_dir --criterion auxiliarycriterion --task auxiliary_translation_task --arch transformer_wmt_en_de1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --stop-min-lr 1e-09 \
        --weight-decay 0.0 --label-smoothing 0.1 \
        --max-tokens 2048 --no-progress-bar --max-update 150000 \
        --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 30 --save-interval 10000 --seed 1111 \
        --ddp-backend no_c10d \
        --dropout 0.3 \
        --patience=20 \
        --reg-alpha 5 \
        --fp16 \
        -s en -t de --save-dir $model_dir \
        --mask-loss-weight 0.03 > regularization.log 2>&1 &
```


