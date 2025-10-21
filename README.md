# STG-GAPE
STG-GAPE: A Parallel Heterogeneous Convolution and Gated Recurrent Synergistic Method for Spiking Time Series Prediction


## Related Papers
* Efficient and Effective Time-Series Forecasting with Spiking Neural Networks, [ICML 2024], (https://arxiv.org/pdf/2402.01533).
* Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators, [NeurIPS 2024], (https://arxiv.org/pdf/2405.14362).


## Installation
To install SeqSNN in a new conda environment:
```
conda create -n SeqSNN python=[3.8, 3.9, 3.10]
conda activate SeqSNN
git clone https://github.com/microsoft/SeqSNN/
cd SeqSNN
pip install .
```

If you would like to make changes and run your experiments, use:

`pip install -e .`

## Training
Take the `iSpikformer` model as an example:

`python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml`

You can change the `yml` configuration files as you want.

You can add, remove, or modify your model architecture in `SeqSNN/network/XXX.py`.

## Datasets

Metr-la and Pems-bay are available at [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g).
Solar and Electricity can be downloaded from  (https://github.com/laiguokun/multivariate-time-series-data).

The folder structure of this project is as follows:
```
SeqSNN
│   README.md 
│   ...
│
└───data
│   │   metr-la.h5
│   │   pems-bay.h5
│   │
│   └───solar-energy
│   │   │   solar_AL.txt
│   │   │   ...
│   │   
│   └───electricity
│   │   │   electricity.txt
│   │   │   ...
│   │   
│   └───traffic
│   │   │   traffic.txt
│   │   │   ...
│
└───forecaster
│   │   ...
│
└───exp
│   │   ...
│
└───outputs
│   │   ...
│
```
You can change the path of the data file in `exp/forecast/dataset/XXX.yml` configuration files.
## Acknowledgement
This project is built upon [forecaster](https://github.com/Arthur-Null/SRD), a general time-series forecasting library.  
We sincerely thank @rk2900 and @Arthur-Null for their initial contributions.

## Contributing
Contributions and suggestions are welcome!  
Feel free to submit issues or pull requests to help improve this project.
subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
