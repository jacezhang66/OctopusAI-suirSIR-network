# Sinking into water without seeing water: semi-supervised underwater image enhancement based on simulated re-degradation process
## Ruihang Zhang, Sen Lin, Yupeng Li
![fishV](https://github.com/jacezhang66/OctopusAI-suirSIR-network/blob/main/fig1/fishV.gif)
<p align="center">Video 1. Enhanced video sample1(Please wait a few seconds to load the video).</p>

## Introduction
The abstract and detailed paper will be accessible subsequent to the manuscript's acceptance.

![structfig](https://github.com/jacezhang66/OctopusAI-suirSIR-network/blob/main/fig1/structfig.jpg)

<p align="center">Figure 1. An overview of our approach.</p>


## Dependencies

- Ubuntu==22.04
- Pytorch==1.12.1
- CUDA==11.1
- OpenCV==3.5.6

## Datasets Preparation


First, split your paired datasets into training, validation, and testing sets.

Then, place the datasets in the `data` folder.

Finally, the structure of  `data`  are aligned as follows: 

```
data
├── labled1
│   ├── input
│   └── GT
│   └── T
├── unlabeled
│   ├── input
│   └── candidate
└── val
    ├── input
    └── GT
└── test
    ├── input
```

## Train

*Set train parameters in the `option.py`.*

 ```shell
 python main.py 
 ```

## Test

Trained_models are available at baidudrive: https://pan.baidu.com/s/1vDB4NM7Ygpv5Ja9W7JHoXg  with code: `aafk`

*Put  models in the `trained_models/`folder.*

*Put your images in `Test_dir/temp_test/input/`*

 ```shell
 python test.py 
 ```

*you can find results from folder `Result_dir/temp_test`.*

## Samples

![fish2V](https://github.com/jacezhang66/OctopusAI-suirSIR-network/blob/main/fig2/fish3V.gif)

<p align="center">Video 2. Enhanced video sample2(Please wait a few seconds to load the video).</p>

![resultfig](https://github.com/jacezhang66/OctopusAI-suirSIR-network/blob/main/fig1/resultfig.jpg)

<p align="center">Figure 2. An overview of our sample2.</p>

## Citation

Further detailed deployment specifics will be disclosed upon acceptance of the manuscript.

