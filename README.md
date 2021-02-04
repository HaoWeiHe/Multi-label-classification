# Multi-lable-classification
Multi-lable-classificationis a project for toxic comment classification.(Take kaggle toxic-comment-classification-challenge as our dataset)


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

- Accurancy trend
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/acc.png" alt="Editor" width="700">
</div>

- Loss trend
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/trend.png" alt="Editor" width="700">
</div>

## Evaluation metrics

### 1.2
Using LSTM with dropout. Accuarncy got 95.7%
```
998/998 [==============================] - 8s 8ms/step - loss: 0.0951 - acc: 0.9571
Loss: 0.09514936059713364
Test Accuracy: 0.9571361541748047
```
Confusion matrix
```
>> training set 

               precision    recall  f1-score   support

        toxic       0.87      0.73      0.80     12275
 severe_toxic       0.51      0.42      0.46      1294
      obscene       0.87      0.74      0.80      6811
       threat       0.50      0.00      0.01       375
       insult       0.76      0.64      0.70      6346
identity_hate       0.78      0.01      0.01      1136

    micro avg       0.83      0.66      0.74     28237
    macro avg       0.71      0.42      0.46     28237
 weighted avg       0.82      0.66      0.72     28237
  samples avg       0.06      0.06      0.06     28237
 
 
 >> testing set 

               precision    recall  f1-score   support

        toxic       0.85      0.70      0.76      3019
 severe_toxic       0.48      0.42      0.44       301
      obscene       0.82      0.71      0.76      1638
       threat       0.00      0.00      0.00       103
       insult       0.73      0.61      0.66      1531
identity_hate       0.00      0.00      0.00       269

    micro avg       0.79      0.63      0.70      6861
    macro avg       0.48      0.40      0.44      6861
 weighted avg       0.75      0.63      0.69      6861
  samples avg       0.06      0.06      0.05      6861
```
