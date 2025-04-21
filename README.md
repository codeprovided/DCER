Source code of DCER.
# 1. Functional Overview of the Code
run.py is a training and testing script for a Multi-Time-Slice Graph Embedding (DCER) model, which is used for user-item rating prediction in recommendation systems. It predicts user ratings for items by aggregating historical interaction information of users and items and combining the embedding representations of users and items. The model also takes into account user curiosity and item novelty to provide more accurate recommendations.
# 2. Environment Preparation
python 3.6

pytorch 1.10.2

cuda 10.1
# 3. Dataset Preparation

The dataset needs to be preprocessed and saved as a .pkl file. The code uses the data stored in the ../dataset/ele2013/t/ folder.


# 4. Model Parameter File Preparation
The pre-trained model parameter file (such as dcer_p_yelp.pt) needs to be downloaded in advance and placed in the ../model/ folder.

# 5. Running the Script
Navigate to the directory where the script is located:

cd /path/to/DCER

Run the script:

python run.py

Adjust the hyperparameters:

parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training')

parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
# 6. Model Input and Output
Input

Dataset: User, item, and rating data, stored as .pkl files.

Model Parameters: Pre-trained model parameter files (such as dcer_p_yelp.pt), stored in the ../model/ folder.

Device: Automatically detects the availability of a GPU and prioritizes using the GPU for computation.

Output

Training Process: During training, the current loss value and the best RMSE/MAE will be printed every 100 steps.
[1,0] loss: 1.234, The best rmse/mae: 1.2345 / 0.9876

Testing Results: After each epoch, the script evaluates the model performance on the test set and outputs the RMSE and MAE.
rmse: 0.9876, mae: 0.7654

Model Saving: If the performance of the current epoch is better than the previous best performance, the model parameters will be saved to the file ../model/dcer_test.pt.


Visualization of Results: The changes in loss during the training process will be visualized and displayed as a chart.

# 7. Code Execution Logic After Running the run.py Script
7.1 Parameter Parsing

Use the argparse module to parse command-line arguments and load model hyperparameters.

7.2 Data Loading

Load the training and testing datasets using pickle.
Wrap the data into TensorDataset and use DataLoader for batch loading.

7.3 Model Initialization

Initialize the embedding layers for users and items.
Initialize the aggregators (UV_Aggregator) and encoders (UV_Encoder) for users and items.
Construct the DCER model.

7.4 Training Process

Conduct multiple rounds (epochs) of training on the training set.
During each epoch, the model calculates the loss and performs backpropagation to update the parameters.
Print the loss value every 100 steps.

7.5 Testing Process

Evaluate the model performance on the test set, calculating RMSE and MAE.
If the current performance is better than the previous best performance, save the model.

7.6 Result Output

Print the RMSE and MAE for each epoch.
Visualize the changes in loss during the training process.

# 8. Dependencies
PyTorch: For building and training neural networks.

NumPy: For numerical computations.

Pandas: For data processing.

Scikit-learn: For calculating RMSE and MAE.

Matplotlib: For visualizing the changes in loss during the training process.

# 9.Notes
Ensure that the data paths are correct and the data format meets the requirements.

If using a GPU, ensure that PyTorch is properly configured with CUDA.

Hyperparameter tuning may need to be optimized based on the specific dataset.

By following the above steps and instructions, you can successfully run the `run.py` file and complete the user-item rating prediction task.

# Description
> data_load: datasets preprocessing
>
>dataset: amazon electric, instrument and yelp 
>
>exp: experiment result
>
> explaination recommendation interpretation metrics calculation，include dr_1、dr_2、dr_3
>
> DCER/args.py hyperparams setting
>
> DCER/run.py main running function
>
> DCER/UV_Aggregators.py and UV_Encoders.py   aggregation functions
=======
Source code of DCER.
# Paper code reading and Reproduce
If you are interested in learning the module structure or coding style, please clone the source code we provided at this github repository. We offer the complete paper code at this repository. If you want to make a quick start and further reproduce the experiment result, please download the ./dataset part and ./model part which is available at kaggle dataset: "dataset and model_produced of DCER" or just search on kaggle with username "codeprovided". Then put the ./dataset and ./model files into ./DCER-master folder and run the ./DCER/run.py.

# Requirements
python 3.6

pytorch 1.10.2

cuda 10.1