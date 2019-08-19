# NSGA-Net
Code accompanying the paper. All codes assume running from root directory. Please update the sys path at the beginning of the codes before running.
> [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://arxiv.org/abs/1810.03522)
>
> Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman and Wolfgang Banzhaf
>
> *arXiv:1810.03522*

![overview](https://github.com/ianwhale/nsga-net/blob/beta/img/overview_redraw.png  "Overview of NSGA-Net")

## Requirements
``` 
Python >= 3.6.8, PyTorch >= 1.0.1.post2, torchvision >= 0.2.2, pymoo == 0.3.0
```

## Results on CIFAR-10
![cifar10_pareto](https://github.com/ianwhale/nsga-net/blob/master/img/cifar10.png  "cifar10")

## Pretrained models on CIFAR-10
The easiest way to get started is to evaluate our pretrained NSGA-Net models.

#### Macro search space ([NSGA-Net-macro](https://drive.google.com/file/d/173_CXA_YbEjg1_Lnfg6vqweTRDiuDi0J/view?usp=sharing))
![macro_architecture](https://github.com/ianwhale/nsga-net/blob/beta/img/encoding.png  "architecture")
``` shell
python validation/test.py --net_type macro --model_path weights.pt
```
- Expected result: *3.73%* test error rate with *3.37M* model parameters, *1240M* Multiply-Adds.

#### Micro search space
![micro_architecture](https://github.com/ianwhale/nsga-net/blob/beta/img/cells.png  "Normal&Reduction Cells")
``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 26 --filter_increment 4 --SE --auxiliary --model_path weights.pt
```
- Expected result: *2.43%* test error rate with *1.97M* model parameters, *417M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1JvMkT1eo6JegtUvT-5qY4LK3xgq-k-OH)). 

``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 34 --filter_increment 4 --auxiliary --model_path weights.pt
```
- Expected result: *2.22%* test error rate with *2.20M* model parameters, *550M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1it_aFoez-U7SkxSuRPYWDVFg8kZwE7E7)). 

``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 36 --filter_increment 6 --SE --auxiliary --model_path weights.pt
```
- Expected result: *2.02%* test error rate with *4.05M* model parameters, *817M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1kLXzKxQ7dazjmANTvgSoeMPHWwYKiOtm)). 

## Pretrained models on CIFAR-100
``` shell
python validation/test.py --task cifar100 --net_type micro --arch NSGANet --init_channels 36 --filter_increment 6 --SE --auxiliary --model_path weights.pt
```
- Expected result: *14.42%* test error rate with *4.1M* model parameters, *817M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1CMtSg1l2V5p0HcRxtBsD8syayTtS9QAu)). 

## Architecture validation
To validate the results by training from scratch, run
``` 
# architecture found from macro search space
python validation/train.py --net_type macro --cutout --batch_size 128 --epochs 350 
# architecture found from micro search space
python validation/train.py --net_type micro --arch NSGANet --layers 20 --init_channels 34 --filter_increment 4  --cutout --auxiliary --batch_size 96 --droprate 0.2 --SE --epochs 600
```
You may need to adjust the batch_size depending on your GPU memory. 

For customized macro search space architectures, change `genome` and `channels` option in `train.py`. 

For customized micro search space architectures, specify your architecture in `models/micro_genotypes.py` and use `--arch` flag to pass the name. 


## Architecture search 
To run architecture search:
``` shell
# macro search space
python search/evolution_search.py --search_space macro --init_channels 32 --n_gens 30
# micro search space
python search/evolution_search.py --search_space micro --init_channels 16 --layers 8 --epochs 20 --n_offspring 20 --n_gens 30
```
Pareto Front               |  Network                  
:-------------------------:|:-------------------------:
![](https://github.com/ianwhale/nsga-net/blob/beta/img/pf_macro.gif)  |  ![](https://github.com/ianwhale/nsga-net/blob/beta/img/macro_network.gif)

Pareto Front               |  Normal Cell              | Reduction Cell
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ianwhale/nsga-net/blob/beta/img/pf_micro.gif)  |  ![](https://github.com/ianwhale/nsga-net/blob/beta/img/nd_normal_cell.gif)  |  ![](https://github.com/ianwhale/nsga-net/blob/beta/img/nd_reduce_cell.gif)

If you would like to run asynchronous and parallelize each architecture's back-propagation training, set `--n_offspring` to `1`. The algorithm will run in *steady-state* mode, in which the population is updated as soon as one new architecture candidate is evaludated. It works reasonably well in single-objective case, a similar strategy is used in [here](https://arxiv.org/abs/1802.01548).  

## Visualization
To visualize the architectures:
``` shell
python visualization/macro_visualize.py NSGANet            # macro search space architectures
python visualization/micro_visualize.py NSGANet            # micro search space architectures
```
For customized architecture, first define the architecture in `models/*_genotypes.py`, then substitute `NSGANet` with the name of your customized architecture. 

## Citations
If you find the code useful for your research, please consider citing our works
``` 
@article{nsganet,
  title={NSGA-NET: a multi-objective genetic algorithm for neural architecture search},
  author={Lu, Zhichao and Whalen, Ian and Boddeti, Vishnu and Dhebar, Yashesh and Deb, Kalyanmoy and Goodman, Erik and  Banzhaf, Wolfgang},
  booktitle={GECCO-2019},
  year={2018}
}
```

## Acknowledgement 
Code heavily inspired and modified from [pymoo](https://github.com/msu-coinlab/pymoo), [DARTS](https://github.com/quark0/darts#requirements) and [pytorch-cifar10](https://github.com/kuangliu/pytorch-cifar). 
