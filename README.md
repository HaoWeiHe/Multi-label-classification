#  Multi-label-classification 
Multi-label-classificationis a project for toxic comment classification. This repository provide moudle/api which was made by refined bert and  expore different models to solve multi-label problem using static word embedding and contextual word representation as input features in different models. 

## Preparing Dependencies
* conda env create -f freeze.yml
* Download refined bert model from this project and move it to model the folder (It may take a while)
```
wget "https://drive.google.com/u/0/uc?id=1PEpcLfhs18NzQKvUYVzcmn-jnnnyXBHz&export=download" -O "bert_classifier.dict"
mv bert_classifier.dict model
```

## Usage
Get the prediction
```
from violation import predict
text = "fuckkkkk u"  
output = predict(text)
print(output)
# output :  {'toxic': 1.0, 'severe_toxic': 0.0, 'obscene': 1.0, 'threat': 0.0, 'insult': 1.0, 'identity_hate': 0.0}
 ```

Get the probability
```
from violation import predict
text = "fuckkkkk u"  
output = predict(text, get_probs = True)
#output: {'toxic': 0.9909837245941162, 'severe_toxic': 0.4319310486316681, 'obscene': 0.9577020406723022, 'threat': 0.08440539240837097, 'insult': 0.884278416633606, 'identity_hate': 0.11709830909967422}
 ```
## Demo
[Online Demo](http://www.haoweihohoho.com/multiLabelDemo)

## Data Resource
The data resource we used to train our model. (Ignore this selection for simply using api )
* [Toxic comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)
* [GloVe word embedding](https://nlp.stanford.edu/projects/glove/) (use for transfer learning)
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## What's New

### API
* Finish bert reconstruction for `Module`
* Finsih Multi-label `Probability`
* Finish -label `Prediction`

### Model 4.1
* Use different representation approach 
* Texual word representation -  `Bert ` /w  ` BCEWithLogitsLoss`

### Model 4.0 
* Aim to improve data imbalance situation (use tech of loss function revised)
* Transfer-learning - `stastical word representation` + `LSTM` /w `Focal Loss` 

### Model 3.0
* Transfer-learning - `word representation` + `LSTM` /w `multiple output layers`
* Establish & tuning `LSTM` model.

### Model 2.0 (Model for Multi-class classification, not Multi-label classification)
* Transfer-learning - `word representation` + `LSTM` /w `single output layer but have multiple neurons` 
* Establish & tuning `LSTM` model.

## Problem Description
You’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. You’ll be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful. 
* See also - [challenge overview](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Different between multi-label and multi-class
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-label-classification/blob/main/Img/multi-label.png" alt="Editor" width="700">
</div>
The picture shows the different between multi-label and multi-class problem. This picture was credited by EurLex.


A multi-class task assume that one picture only have one class at a time, while a milti-label task assume that one picture could have serveral classes. In our task, every toxic comment could have serveral toxicity types at the same time. (For instance, a comment could have both "threat" and "insult").

In the task is to predict the probability of different toxicity types for a given comment. So, essentially, our models are going to predict the properbility of insult, threat and toxic etc. i.e., P(insult), P(threat) and P(toxic). Looking at the training data, we knoe that this is a classification problem where we want to maximize area under the curve(AUC) and check the precision/recall/f1, since it have imbalanced data issue.

## Model options

There are serveral models to pridict toxic types using dense features. Such as logistic regression, tree-base models or deep learning. 
## Logistic regression
One of the advantage of logistic regression is resonable for effiecent to train and predict. (It only takes O(n) time complexity). In additional, it also extremely interpretable. However, A major limitation of the linear model is that it assumes linearity exists between the input features and prediction.

## Tree based models
Another modeling options are tree beased models with dense features. Trees are effiecent, interpreatable and  able to utilize non-linear relations between features that aren't avaliable to Logistic regression. However, the training time-complexity is higher than logistic regression.

## Deep learning model
With sufficient computational power and the amount of data, deep learning can be very powerful in predicting comment types. Training the model as well as evaluation could be very expensive, but we still have some tech to ease this, such as multi-task tech. In this repostory, we will focus on this technique.

### Seperate neutral network
One way to train seperate NN for each toxicity types of the P(insult), P(threat) and P(toxic). However, since the training / evaluation time could be slow and require high computaional power. The multi-task tech as following introduce will ease this problem.

### Muti-task neutral network
Another way to to detect different types of of toxicity comments is muti-task NN. We've noticed that to predict the probability of different toxicity comment, such as insult, threat and toxic are similar task as the similar input with word representation. If we are tring to detect insult, it would be helpful for a model to know a comment has a threat context as sharing knowledge. Hence, we can try a NN model with shared layers (for the sharing knowledge) and appended with specific layers for each task's prediction. Thus, these tasks could use the same weight of shared layers. But learns the information of each specific knowledge to the taks by sepcific layers. 

<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-label-classification/blob/main/Img/Multitask%20Learning.png" alt="Editor" width="700">
</div>  
A Mutli-task flowchart credited by the authors of this paper - Context-Aware Human Activity and Smartphone Position-Mining with Motion Sensors.

### Bidirectional Encoder Representations from Transformers (BERT)
To tackle this task, we also can leverage transfer learning tech and the idea of multi-task learning we just learned.
BERT is a transformer encoder and  also a language representation model. Unlike other language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.  As a result, the pre-train Bert model can be fine tuned by just one additional output layer and create SOTA achievement on a wide range of applications. 

In this Github repository we use pre-train Bert with additional layers to address the multi-label problem. For more detail of layers information, please check out the subsection below - Model 4.1 - bert multi-label refine.


### Multi-class neutral network
Some times, a product team might want to shift its focus to predict only one certain toxice type. In this case, we can take this taks as a multi-class classifion problem using a single output layer /w multiple neurons. (Noted that multi-class is defferent from multi-label which we've introduced in the above section.)

## Loss function
To deal with multi-class problem, we use `focal loss` as our loss fuction (Model 4.0). Because in our dataset, the data is extremely unbalance. To tackle this question, choosing a suitable loss function is one of the answer. Focal Loss is an improved version of Cross-Entropy Loss that tries to handle the class imbalance problem by assigning more weights to hard or easily misclassified examples and to down-weight easy examples 

So Focal Loss reduces the loss contribution from easy examples and increases the importance of correcting misclassified examples.
[See more](https://arxiv.org/ftp/arxiv/papers/2006/2006.01413.pdf)

## Observation

- What does the data look like?
<div align="center">
    <img src="https://github.com/HaoWeiHe/multi-lable-classification/blob/main/Img/data_sample.png" alt="Editor" width="700">
</div>


- Generate descriptive statistics
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/categories.png" alt="Editor" width="700">
</div>

- Total comment counts for different labels
<div align="center">
    <img src="https://github.com/HaoWeiHe/multi-lable-classification/blob/main/Img/commentCounts.png" alt="Editor" width="500">
</div>
A quick calculation : sum(label_counts)/num_of_sample = 35098/159571 = 0.219, which indicate that the lower bound(accurancy metric) is around 78.1%


- Count numbers of different categories (Training set)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/mulit-label-count.png" alt="Editor" width="700">
</div>

- Count numbers of different categories (Testing set before data restructure)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/test_ori.png" alt="Editor" width="700">
</div>


- Count numbers of different categories (Testing set after data constructure)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/test.png" alt="Editor" width="700">
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

## Data Restructure
- Training set - all data from train.csv
- Testing set - all data from test.csv except the data with value of -1, which which means it was not used for scoring

## Model Strucutre
- Model 4.1 - bert mulit-label refine 
```
MultiLabel(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        # 12 BertLayers
        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=6, bias=True)
)
```
- Model 3.0 -multiple output layers
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/structure2.png" alt="Editor" width="900">
</div>


- Model 2.0 - single output layer /w multiple neurons (Model for Multi-label classification, not Multi-label)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/structure1.png" alt="Editor" width="400">
</div>

## Training 


- Learning Curve - Acc (Model 4.0)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/acc_foss.png" alt="Editor" width="500">
</div>

- Learning Curve - Loss (Model 4.0)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/loss%20foss.png" alt="Editor" width="500">
</div>

- Learning Curve - AUC-ROC (Model 4.0)
<div align="center">
    <img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/auc_foss.png" alt="Editor" width="500">
</div>


## Evaluation Metrics

### Model 4.0

Use the Focal Loss. Accuarncy got 0.9934 and AUC got 0.97. (Auc of Model 2.0 got 0.955)
(This loss function aim to improve data imbalance situation)


```
2000/2000 [==============================] - 18s 9ms/step - loss: 0.0493 - acc: 0.9934 - auc: 0.9700
Loss: 0.04926449805498123
Test Accuracy: 0.9933727383613586

```
Confuse Matric
```
>> training set 

               precision    recall  f1-score   support

        toxic       0.94      0.65      0.76     15294
 severe_toxic       0.96      0.05      0.10      1595
      obscene       0.94      0.60      0.73      8449
       threat       1.00      0.01      0.03       478
       insult       0.89      0.37      0.52      7877
identity_hate       0.92      0.01      0.02      1405

    micro avg       0.93      0.51      0.66     35098
    macro avg       0.94      0.28      0.36     35098
 weighted avg       0.93      0.51      0.63     35098
  samples avg       0.06      0.04      0.05     35098


>> testing set 

               precision    recall  f1-score   support

        toxic       0.65      0.72      0.68      6090
 severe_toxic       0.44      0.06      0.10       367
      obscene       0.83      0.53      0.65      3691
       threat       0.33      0.00      0.01       211
       insult       0.81      0.30      0.44      3427
identity_hate       0.40      0.00      0.01       712

    micro avg       0.71      0.51      0.59     14498
    macro avg       0.58      0.27      0.31     14498
 weighted avg       0.71      0.51      0.56     14498
  samples avg       0.07      0.05      0.05     14498

```
### Model 3.0

Using LSTM with dropout. Accuarncy got only 18.4% (Overfitting by observing the historical curve.)

```
2000/2000 [==============================] - 23s 11ms/step - loss: 0.5070 - dense_8_loss: 0.2167 - dense_9_loss: 0.0189 - dense_10_loss: 0.1153 - dense_11_loss: 0.0150 - dense_12_loss: 0.1050 - dense_13_loss: 0.0361 - dense_8_acc: 0.9138 - dense_9_acc: 0.9932 - dense_10_acc: 0.9561 - dense_11_acc: 0.9966 - dense_12_acc: 0.9598 - dense_13_acc: 0.9893
Test Score: 0.5070158839225769
Test Accuracy: 0.21666644513607025

```

### Model 2.0

Using LSTM with dropout. Accuarncy got 99.7% (The result is over optimism, for we use argmax to evaluate). AUC got 0.9559.

```
2000/2000 [==============================] - 18s 9ms/step - loss: 0.0872 - acc: 0.9965 - auc: 0.9559
Loss: 0.0871838703751564
Test Accuracy: 0.9964519143104553
```
Confusion matrix. (To look into each categories)
```
>> training set 

               precision    recall  f1-score   support

        toxic       0.88      0.78      0.82     15294
 severe_toxic       0.58      0.37      0.45      1595
      obscene       0.88      0.74      0.81      8449
       threat       0.25      0.00      0.00       478
       insult       0.77      0.64      0.70      7877
identity_hate       0.67      0.00      0.01      1405

    micro avg       0.84      0.68      0.75     35098
    macro avg       0.67      0.42      0.47     35098
 weighted avg       0.82      0.68      0.73     35098
  samples avg       0.07      0.06      0.06     35098
  >> testing set 

               precision    recall  f1-score   support

        toxic       0.55      0.82      0.66      6090
 severe_toxic       0.39      0.42      0.40       367
      obscene       0.70      0.69      0.69      3691
       threat       0.00      0.00      0.00       211
       insult       0.62      0.58      0.60      3427
identity_hate       0.40      0.00      0.01       712

    micro avg       0.59      0.67      0.63     14498
    macro avg       0.44      0.42      0.39     14498
 weighted avg       0.58      0.67      0.60     14498
  samples avg       0.07      0.06      0.06     14498
  
```

 
## Some Resource
* [loss function](https://neptune.ai/blog/keras-loss-functions) - (It's fruitful!!)
