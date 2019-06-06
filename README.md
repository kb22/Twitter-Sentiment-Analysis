# Twitter Sentiment Analysis using Neural Networks
The repo includes code to process text, engineer features and perform sentiment analysis using Neural Networks.

## Setup

### Install python
1. Install pyenv for managing Python versions
```
brew install pyenv
```
2. Install python with this flag
```
CFLAGS="-I$(xcrun --show-sdk-path)/usr/include" pyenv install 3.7.2
```

### Get the code
1. Clone the repo to your machine
```
git clone https://github.com/kb22/Twitter-Sentiment-Analysis-using-Neural-Networks.git
```
2. Move into the folder
```
cd Twitter-Sentiment-Analysis-using-Neural-Networks
```
3. Install all dependencies
```
pip install -r requirements.txt
```

### Download the dataset
The dataset has been taken from [Kaggle](https://www.kaggle.com/kazanova/sentiment140)
1. Download the file from kaggle.
2. Extract the zip and rename the `csv` to `dataset.csv`
3. Create a folder `data` inside `Twitter-Sentiment-Analysis-using-Neural-Networks` folder
3. Copy the file dataset.csv to inside the `data` folder

## Working the code

### Understanding the data
The Jupyter notebook **Dataset analysis.ipynb** includes analysis for the various columns in the dataset and a basic overview of the dataset.
1. Run Jupyter
```
jupyter notebook
```
2. Select the file **Dataset analysis.ipynb** from the list



