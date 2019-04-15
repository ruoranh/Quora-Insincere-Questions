# Quora Insincere Questions – Kaggle Competition #

“An existential problem for any major website today is how to handle toxic and divisive content” – Quora.com

Websites have long struggled to deal with negative content posted by users, the sheer volume of data being generated makes it impossible to use manual review to solve the issue, machine learning algorithm is the obvious answer. However, the complexity and ambiguity of natural human language makes this a particularly difficult problem to solve and an area of intense studying in the science data community.

In this project I used Natural Language Processing and Supervised Machine Learning to distinguish between sincere and insincere questions supplied by Quora.com. I employed a variety of techniques such as Bag of Words, LSTM neural network, Word2Vec, pipelines and GridSearch to try and achieve the best predictive result. Currently Quora use a combination of machine learning and manual review.


## Data Overview ##
Quora provided the data set which can be found at, https://www.kaggle.com/c/quora-insincere-questions-classification. The data set contained 1.31 million questions, pre-classified as either sincere or insincere, of these questions, 6.2% were classified as insincere. 
This data set brings about 2 challenges: 

1. large size of the data, which resulted in very long computation times. 

2. extreme imbalance between classes, baseline accuracy of 93.8% make accuracy a very poor measure of model performance, so F1 score was used instead. While less intuitive, F1 is a much better measure, because it looks at not only what the model got right but also what it got wrong.


## EDA (Workbook 01) ##
Trying to get a sense of the dataset, looking for any patterns that may help with classification. Investigated baseline accuracy, question length distribution, 100 most common words and bigrams based on classification.  

The most interesting discovery was the difference in common words/bigrams between classifications. After removing common stop words. Sincere questions contained mostly words associated with questions, such as ‘one’, ‘best’, ‘think’, ‘work’ etc, with the exception of the word ‘India’, this is because India is one of the largest user base for Quora. Where as insincere questions contain mostly words associated with identity ‘Muslim’, ‘American’, ‘women’, ‘men’, ‘liberal’ etc, with ‘trump’ also being a top word. Bigrams showed a very similar pattern. This provide a very good indication of what type of questions are featured in the insincere set.

![](README%20Images/Word%20Cloud.png)

## Basic Text Modelling (Workbook 02) ##
Bag of words vectorizer and Native Bayes. Train/Test split 75/25, stratified. Cleaning involved removed non-alpha characters, removed standard NLTK stopwords, lower case all characters.

In total seven different Model were created using the below methods:

**Raw Tokens** – Vectorize on all single words on raw data without any cleaning. This was created as a means of creating a baseline accuracy for F1 score, as I expected this to be the lowest scoring model.

**Tokens** – Vectorize on all single words on cleaned data, this show the effects of cleaning, useful for trying different cleaning methods, e.g. different stop word lists.

**All Bigrams** – Vectorize on all single words and bigrams on cleaned data. Should create lots of noise, but useful for comparison against the best bigrams as determined using various association measures. 

**All Trigrams** – Vectorize on all single words, bigrams and trigrams, on cleaned data. Similar to All Bigrams, should create lots of noise, but useful for comparison against the best trigrams as determined using various association measures. 

**TFIDF** – TFIDF weight assigned to all single words on cleaned data. Instead of simply counting how many times the word appears in a particular question, it also looks at how many questions the word appears in. The effect is that words that don’t appear in many questions but appears frequently in this question will have a higher weight.

**N Bigrams** – 80 best bigrams found using ‘Likelihood Ratio’ and 200 best bigrams using ‘Point Mutual Information’ were combined into a unique list. These bigrams in the cleaned data were replaced using a single token (i.e. ‘united states’ become ‘united_states’) so the processed question can be vectorized as single words and signals in these bigrams will not be duplicated. Other number of bigrams using each method were tried, these seem to produce the best results.

**N Trigrams** – 40 best trigrams found  using ‘Likelihood Ratio’ and 40 best bigrams using ‘Point Mutual Information’ were found and combined into a unique list, then integrated into the questions using the same method as N Bigrams


![](README%20Images/Basic_score.png)

Looking at the accuracy chart for this set of models, you can see that most of the results were above baseline accuracy and all other models achieved higher accuracy than Raw Token. Simply looking at accuracy scores, it would appear that the models are achieving reasonable results.  However, the F1 scores tells a very different story, every model produced a lower F1 than Raw Token. The most extreme of which is the TFIDF model, looking at the confusion matrix below shows us why.

![](README%20Images/TFIDF_confusion.png)

The TFIDF model predicted almost everything as sincere, but it was still able to achieve above baseline accuracy due to the small amount of insincere questions it got right. However, the F1 score was able to capture the fact it got almost all of the insincere questions wrong, correctly giving it a very low F1 score of 0.15. This confirm that F1 score is a better measure of performance for this scenario.

Random Forest was also trialled but resulted in much longer computation time without boost in accuracy.


## Lemmatization (Workbook 03) ##

In this step, I lemmatized the data with Part of Speech tagging, by using the spaCy library. Then performed all the same modelling as the previous step, still comparing against Raw Token model.

![](README%20Images/Lemmatized_score.png)

Results showed a similar pattern to the un-lemmatized data. However, there was a slight increase in F1 score of All Bigram and All Trigram models, but a slight decreased F1 score for other models. This is because, while lemmatization removed some of the noise in the data, it also removed some of the signal. Resulting in a slight boost to the performance of noisier models like All Bi/Trigram, but reduced the performance of models which had built in feature selection like TFIDF and Best Bi/Trigrams. However, Raw Token model still has the best F1 score.


## Identify Label (Workbook 04) ##
Having tried the standard bag of words techniques I attempted to increase model performance through feature engineering. As discussed earlier, there was a clear pattern of identity words associated with insincere questions, by replacing common identity words with a single label, the model will consider all these words to have the same meaning, which should increase model performance. As such, I replaced all religious, racial and nationality words with their own respective group label, then re-trained all models, except Raw Token.

![](README%20Images/Labelled_score.png)

The new feature increased F1 score on all models except All Trigram and TFIDF. The biggest gain was in All Bigram which increased by almost 0.1. Unfortunately, Raw Token still has the highest F1 score.


## Long Short Term Memory (Workbook 05) ##
With the limited success of bag of words methods, Long Short Term Memory (LSTM) was the logical next step. The embedding process can derive complex meanings for each word, based on what words they often appear next to in the corpus and store these meaning in a multiple dimensional vector. Therefore, the meaning assigned to each word is less binary when compared to bag of words techniques, allowing for more complex interpretation of semantics. Furthermore, LSTM has the ability to assign weights to each word beyond the word meaning from the embedding process. Through an iterative backpropagation process, LSTM is able to feature select, by reducing the weight on words that represent noise and increase weights on words that contain lots of signal.

### 1D convolutional Layer LSTM ###
Initial run of LSTM model showed promising result, however the model took over 6 hours to train. To reduce training time, I employed a single 1D convolutional layer as a feature reduction technique. This reduced the training time to around 2-3 hours, I was able to train a number of these models by varying the following parameters.

![](README%20Images/Conv_LSTM_parameters.png)

![](README%20Images/Conv_LSTM_score.png)

The best results were achieved using large embedding vocabulary, high embedding dimension and low drop rate. Again, none of the models were able to outperform the Raw Token Naïve Bayes model.   

### LSTM only ###
It seems that the convolutional layer was removing too much signal from the data. Back to training a LSTM only model. Due to the large computation time of these models I was not able to fully explore the tunning parameters, but only had time to train 3 models plus a model using a google pre-trained Word2Vec embedding.


![](README%20Images/LSTM_score.png)
 
Finally, these models achieve results better than Raw Token Naïve Bayes. However, if we look at the confusion matrix below, the model is still mislabelling 44% of insincere questions. This isn’t a good outcome.

![](README%20Images/LSTM_confusion.png)


## Focus on the objective (Workbook 05) ##

At this point I asked myself, what is the objective of this model. Is it to capture as many insincere questions as possible, so user of the Quora site will not be exposed to such content? Or is it to minimise the number of questions being captured by the model, so as to minimise the amount of content Quora admins needs to manually review. I decided for the purpose of this project I will prioritise the capture of insincere questions at the expense of having extra false positives.

As all models produce a probability rather than a yes or no answer, we are able to change the rounding threshold to change the model priority, by default the rounding threshold is set at 0.5.

![](README%20Images/Threshold_chart.png)

The above chart shows that as the threshold decrease, the percentage of insincere questions being correctly labelled increases, however the percentage of sincere questions being incorrectly labelled also increase. The advantage is that the rate of increase in the correctly label insincere questions is much higher than the wrongly labelled sincere questions at anything below a 0.05 threshold.

![](README%20Images/Threshold_table.png)

The final model I chose had a threshold of 0.1, this was able to capture 86% of insincere questions, and only wrongly labelled 8% of sincere questions as insincere. A much better result than the default LSTM model.


## Web App ##
Lastly, I created a simple web app that allow users to ask the model any question and the model will predict whether the question is sincere or not.

![](README%20Images/Web_app.png)


## Summary ##
This project has opened by eyes to the fascinating world of NLP and allowed me to experience both the amazing abilities and the many limitations of the current NLP field. 

My model is able to reduce the amount of questions requiring manual review by 87%, when compared to if no model were used, making the insincere question problem manageable. Given more time, I am confident I could increase the result beyond what I was able to achieve so far. Next steps include combining feature engineering with LSTM models and explore more embedding methods such as Word2Vec and FastText.

