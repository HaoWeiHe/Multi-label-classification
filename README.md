# Multi-lable-classification
Multi-lable-classificationis a project for toxic comment Classification.(Take kaggle toxic-comment-classification-challenge as our dataset)


## Data Resource
* [Toxic comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)
* [GloVe word embedding](https://nlp.stanford.edu/projects/glove/)
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## What's New
1.1 Observation/anlysis the dataset before diving in
1.2  Establish & tuning `LSTM` model.

## Observation
- Generate descriptive statistics
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/categories.png" alt="Editor" width="700">
</div>

- What does the data look like?
<div align="center">
	<img src="https://github.com/HaoWeiHe/multi-lable-classification/blob/main/Img/data_sample.png" alt="Editor" width="700">
</div>


- Total comment counts for different labels
<div align="center">
	<img src="https://github.com/HaoWeiHe/multi-lable-classification/blob/main/Img/commentCounts.png" alt="Editor" width="500">
</div>
A quick calculation : sum(label_counts)/num_of_sample = 35098/159571 = 0.219, which indicate that the lower bound(accurancy metric) is around 78.1%


- Count numbers of different categories
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/mulit-label-count.png" alt="Editor" width="700">
</div>

- Word Length distribution
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/word_frequency_distribution.png" alt="Editor" width="700">
</div>
As per the mean and standard diviation from this data observation, we can set our embedding length to 255 and it can contain around 95% words in a text

- Word colud
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/wordCloud.png" alt="Editor" width="700">
</div>
A glimsp of our high frequency words


## Evaluation metrics

### 2.1
Using LSTM with dropout. Accuarncy got 95.7%
```
998/998 [==============================] - 8s 8ms/step - loss: 0.0951 - acc: 0.9571
Loss: 0.09514936059713364
Test Accuracy: 0.9571361541748047
```
