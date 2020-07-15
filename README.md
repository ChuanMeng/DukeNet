# DukeNet (SIGIR 2020 full paper)
The code for [DukeNet: A Dual Knowledge Interaction Network for Knowledge-Grounded Conversation]()

## Reference
If you use any source code included in this repo in your work, please cite the following paper.
```
@inproceedings{chuanmeng2020dukenet,
 author = {Meng, Chuan and Ren, Pengjie and Chen, Zhumin and Sun, Weiwei and Ren, Zhaochun and Tu, Zhaopeng and de Rijke, Maarten},
 booktitle = {SIGIR},
 title = {DukeNet: A Dual Knowledge Interaction Network for Knowledge-Grounded Conversation},
 year = {2020}
}
```

## Requirements 
* python 3.6
* pytorch 1.2-1.4
* transformers

## Datasets
We use Wizard of Wikipedia and Holl-E datasets. Note that we used modified verion of Holl-E relased by [Kim et al](https://arxiv.org/abs/2002.07510?context=cs.CL).
Both datasets have already been processed into our defined format, which could be directly used by our model.

You can manually download the datasets at [here](https://drive.google.com/drive/folders/1dgkCKaypKHej-NE2HYuiP1VhuF-xCgqT?usp=sharing).

## Running Experiments
Note that it's rather **time-consuming** to train the DukeNet with BERT and dual learning. Therefore we upload our pretrained checkpoints on two datasets, and you can manually download them at [here]().


```
Holl-E_single golden reference
{"F1": 30.32,
 "BLEU-1": 29.97, 
 "BLEU-2": 22.2, 
 "BLEU-3": 20.09, 
 "BLEU-4": 19.15, 
 "ROUGE_1_F1": 36.53, 
 "ROUGE_2_F1": 23.02, 
 "ROUGE_L_F1": 31.46, 
 "Distinct-1": 5.9, 
 "Distinct-2": 23.77, 
 "METEOR": 30.93, 
 "t_acc": 92.11, 
 "s_acc": 30.03}
```
