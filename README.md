# Reduced Glass Temperature Prediction Via PSO-SVR
Simple PSO-SVR model for estimating reduced glass temperature of different alloys solely based on alloy composition.

## Repository Contents
|File Name                                  |File Type|Description
|:------------------------------------------|:--------|:-------------------
|Metallic_Glass_Forming_with_features2.xlsx |Excel    |Columns added for each alloy constituent in original Excel data file along with corresponding percentage values. More details in "Data Attribution" section.
|pso_svr.py                                 |PY       |Contains code for training PSO-SVR model

##


## Getting Started
The following are the procedures for downloading and training the models in this repository using modified data derived from “Machine Learning Materials Datasets”:

1. Download "Metallic_Glass_Forming_with_features2.xlsx" and "pso_svr.py" from this repository into the same directory.
2. Open cmd/terminal and change directory to dataset folder.
3. Install required packages by typing "pip install -r requirements.txt" into cmd.
4. Run "pso_svr.py" by typing "python pso_svr.py" in cmd.

## Results
|Model Parameters        |Value
|:-----------------------|----------------:
|Best C                  |1.35 (3 s.f.)
|Best Epsilon            |0.010 (3 s.f.)
|Best Gamma              |0.334 (3 s.f.)

|Performance Indicators  |Value
|:-----------------------|----------------:
|Total Parameters        |277
|Total Memory (KB)       |101 (3 s.f.)
|Best PSO Score          |0.00110 (3 s.f.)
|RMSE                    |0.0252 (3 s.f.)
|SMAPE                   |3.22 (3 s.f.)
|MAPE                    |3.23 (3 s.f.)

## Data Attribution
This project uses modified data derived from “Machine Learning Materials Datasets” assembled by Vanessa Nilson under the guidance of Prof. Dane Morgan,
available on Figshare: https://figshare.com/articles/dataset/MAST-ML_Education_Datasets/7017254.
The original dataset is licensed under the MIT License.
