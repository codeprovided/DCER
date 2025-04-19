Source code of DCER.
# Paper code reading and Reproduce
If you are interested in learning the module structure or coding style, please clone the source code we provided at this github repository. We offer the complete paper code at this repository. If you want to make a quick start and further reproduce the experiment result, please download the ./dataset part and ./model part which is available at kaggle dataset: "dataset and model_produced of DCER" or just search on kaggle with username "codeprovided". Then put the ./dataset and ./model files into ./DCER-master folder and run the ./MTGE/run.py.

# Requirements
python 3.6

pytorch 1.10.2

cuda 10.1

# Description
> data_load: datasets preprocessing
>
>dataset: amazon electric, instrument and yelp 
>
>exp: experiment result
>
> explaination recommendation interpretation metrics calculation，include dr_1、dr_2、dr_3
>
> MTGE/args.py hyperparams setting
>
> MTGE/run.py main running function
>
> MTGE/UV_Aggregators.py and UV_Encoders.py   aggregation functions
>
