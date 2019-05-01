# NSGA-Net
Code accompanying the paper. All codes assume running from root directory. 
> [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://arxiv.org/abs/1810.03522)
>
> Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman and Wolfgang Banzhaf
>
> *arXiv:1810.03522*

![overview](https://github.com/ianwhale/nsga-net/blob/beta/img/overview_redraw.png  "Overview of NSGA-Net")

### Pretrained models on CIFAR-10
The easist way to get started is to evaluate our pretrained NSGA-Net models.
#### Micro search space ([NSGA-Net (6 @ 424)](https://drive.google.com/file/d/16v60Ex2C2ZNwCFACTEPZJrpVU9x5OWPj/view?usp=sharing))
![micro_architecture](https://github.com/ianwhale/nsga-net/blob/beta/img/cells.png  "Normal&Reduction Cells")
``` shell
python micro_search_space/test.py --arch NSGANet --init_channels 34 --filter_increment 4 --model_path weights.pt
```
- Expected result: *2.62%* test error rate with *2.42M* model parameters, *550M* Multiply-Adds.

#### Macro search space ()
![macro_architecture](https://github.com/ianwhale/nsga-net/blob/beta/img/encoding.png  "architecture")

