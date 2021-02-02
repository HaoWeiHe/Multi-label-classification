# Multi-lable-classification
Multi-lable-classificationis a project for toxic comment Classification.(Take kaggle toxic-comment-classification-challenge as our dataset)


## Data Resource
* [Toxic comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)


## What's New
1.1 Observation/anlysis the dataset before diving in. 

## Observation
- What does the data look like?
<div align="center">
	<img src="https://github.com/HaoWeiHe/multi-lable-classification/blob/main/Img/data_sample.png" alt="Editor" width="700">
</div>


- Total comment counts for different labels.
<div align="center">
	<img src="https://github.com/HaoWeiHe/multi-lable-classification/blob/main/Img/commentCounts.png" alt="Editor" width="700">
</div>

- Word Length distribution
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/word_frequency_distribution.png" alt="Editor" width="700">
</div>
Base on the mean and standard diviation from this data observation, we can set our embedding length to 255, and it can contain around 95% words in a text

- Word colud
<div align="center">
	<img src="https://github.com/HaoWeiHe/Multi-lable-classification/blob/main/Img/wordCloud.png" alt="Editor" width="700">
</div>
A glimsp of our high frequency words

