# Twitter Sentiment Analysis using Neural Networks
The repo includes code to process text, engineer features and perform sentiment analysis using Neural Networks. The project uses **LSTM** to train on the data and achieves a **testing accuracy of 79%**.

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
2. Select the file **Dataset analysis.ipynb** from the list to see dataset analysis.

### Twitter Sentiment Analysis
The whole project is broken into different Python files from splitting the dataset to actually doing sentiment analysis. The steps to carry out Twitter Sentiment Analysis are:
1. Run the file `train-test-split.py` to split the Twitter dataset into training and testing data.
```
python train-test-split.py
```
2. Run the file `preprocessing.py` to process the tweets.
- Remove @user mentions
- Remove non-alphabetic characters + spaces + apostrophe
- Remove links
- Remove single characters
- Remove stopwords
- Lemmatize words
- Stem words
```
python preprocessing.py
```
3. After processing of the tweets, **LSTM** can be used to train on the data and test the accuracy on the test data.
```
python lstm.py
```




