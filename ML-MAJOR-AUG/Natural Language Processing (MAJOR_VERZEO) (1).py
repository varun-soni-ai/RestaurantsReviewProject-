#!/usr/bin/env python
# coding: utf-8

# # • Project Name:
# Machine Learning August Major Project
# ## • Project Description:
# Sentiment analysis studies the subjective information in an expression, that is, the
# opinions, appraisals, emotions, or attitudes towards a topic, person or
# entity. Expressions can be classified as positive, negative, or neutral. For example:
# “I really like the new design of your website!” → Positive.
# ## • You are provided the restaurant review datasets taken from kaggle. Build a
# ## machine learning model
# (using NLP) to predict the label of the review either positive or negative. You
# are supposed to first process the data, then clean it using stemming/
# lemmatization, create a pipeline with Vectorization model and ML algorithm to
# predict the final sentiment.
# • There are two features - ‘review’ - the sentence and ‘sentiment’ - the label for the
# review. 1 means positive review and 0 means negative review.
# ### • Data Set Link: -
# https://drive.google.com/drive/folders/1iSvKkvmoEQbdcZk08UTMt_2Kf0_kNICh?usp=sharing

# # Natural Language Processing
# ## What Is Natural Language Processing?
# Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.
# 
# NLP combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. Together, these technologies enable computers to process human language in the form of text or voice data and to ‘understand’ its full meaning, complete with the speaker or writer’s intent and sentiment.
# 
# NLP drives computer programs that translate text from one language to another, respond to spoken commands, and summarize large volumes of text rapidly—even in real time. There’s a good chance you’ve interacted with NLP in the form of voice-operated GPS systems, digital assistants, speech-to-text dictation software, customer service chatbots, and other consumer conveniences. But NLP also plays a growing role in enterprise solutions that help streamline business operations, increase employee productivity, and simplify mission-critical business processes.
# 
# ## NLP tasks
# Human language is filled with ambiguities that make it incredibly difficult to write software that accurately determines the intended meaning of text or voice data. Homonyms, homophones, sarcasm, idioms, metaphors, grammar and usage exceptions, variations in sentence structure—these just a few of the irregularities of human language that take humans years to learn, but that programmers must teach natural language-driven applications to recognize and understand accurately from the start, if those applications are going to be useful.
# 
# Several NLP tasks break down human text and voice data in ways that help the computer make sense of what it's ingesting. Some of these tasks include the following:
# 
# ### 1. Speech recognition, also called speech-to-text, is the task of reliably converting voice data into text data. Speech recognition is required for any application that follows voice commands or answers spoken questions. What makes speech recognition especially challenging is the way people talk—quickly, slurring words together, with varying emphasis and intonation, in different accents, and often using incorrect grammar.
# ### 2. Part of speech tagging, also called grammatical tagging, is the process of determining the part of speech of a particular word or piece of text based on its use and context. Part of speech identifies ‘make’ as a verb in ‘I can make a paper plane,’ and as a noun in ‘What make of car do you own?’
# ### 3. Word sense disambiguation is the selection of the meaning of a word with multiple meanings  through a process of semantic analysis that determine the word that makes the most sense in the given context. For example, word sense disambiguation helps distinguish the meaning of the verb 'make' in ‘make the grade’ (achieve) vs. ‘make a bet’ (place).
# ### 4. Named entity recognition, or NEM, identifies words or phrases as useful entities. NEM identifies ‘Kentucky’ as a location or ‘Fred’ as a man's name.
# ### 5. Co-reference resolution is the task of identifying if and when two words refer to the same entity. The most common example is determining the person or object to which a certain pronoun refers (e.g., ‘she’ = ‘Mary’),  but it can also involve identifying a metaphor or an idiom in the text  (e.g., an instance in which 'bear' isn't an animal but a large hairy person).
# ### 6. Sentiment analysis attempts to extract subjective qualities—attitudes, emotions, sarcasm, confusion, suspicion—from text.
# ### 7. Natural language generation is sometimes described as the opposite of speech recognition or speech-to-text; it's the task of putting structured information into human language. 
# 
# 

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re     # library to clean data
import nltk    # Natural Language Tool Kit.


# In[24]:


ds = pd.read_csv("C:\\Users\\DELL\\Desktop\\\Restaurant_Reviews.tsv", encoding ='latin-1‪',delimiter = '\t', quoting = 3)
ds.head(9)
# Import dataset with setting delimiter as ‘\t’ as columns are separated as tab space.
#Reviews and their category(0 or 1) are not separated by any other symbol but with tab space as most of the other 
#symbols are is the review (like $ for the price, ….!, etc) 
#and the algorithm might use them as a delimiter, which will lead to strange behavior (like errors, weird output) in output. 
 


# ## Cleaning the texts and Text Cleaning or Preprocessing 
#  
# 
# ### Remove Punctuations, Numbers: Punctuations, Numbers don’t help much in processing the given text, if included, they will just increase the size of a bag of words that we will create as the last step and decrease the efficiency of an algorithm.
# ### Stemming: Take roots of the word 
# ### Convert each word into its lower case: For example, it is useless to have some words in different cases (eg ‘good’ and ‘GOOD’).

# In[25]:




nltk.download('stopwords')

from nltk.corpus import stopwords   # to remove stopword

from nltk.stem.porter import PorterStemmer    # for Stemming propose

# Initialize empty array
# to append clean text
corpus = []

# 1000 (reviews) rows to clean
for i in range(0, 1000):
    # column : "Review", row ith
    review = re.sub('[^a-zA-Z]', ' ', ds['Review'][i])
    # convert all cases to lower cases
    review = review.lower()
    # split to array(default delimiter is " ")
    review = review.split()
    # creating PorterStemmer object to
    # take main stem of each word
    ps = PorterStemmer()
    # loop for stemming each word in string array at ith row
    review = [ps.stem(word) for word in review
         if not word in set(stopwords.words('english'))]
    # rejoin all string array elements
    # to create back into a string
    review = ' '.join(review)
    # append each string to create
    # array of clean text
    corpus.append(review)


# In[26]:


print(corpus)


# ## Creating the Bag of Words model
# ### Tokenization is the process of tokenizing or splitting a string, text into a list of tokens. One can think of token as parts like a word is a token in a sentence, and a sentence is a token in a paragraph.

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()   # X contains corpus (dependent variable)
# y contains answers if review
# is positive or negative
y = ds.iloc[:, -1].values                


# ### Splitting the dataset into the Training set and Test set

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Training the SVM model on the Training set

# In[29]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# ## Fitting Random Forest Classification  to the Training set 

# In[30]:


from sklearn.ensemble import RandomForestClassifier

# n_estimators can be said as number of
# trees, experiment with n_estimators
# to get better results
model = RandomForestClassifier(n_estimators = 501,criterion = 'entropy')
model.fit(X_train, y_train)


# ## Predicting the Test set results

# In[31]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[32]:


# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred


# ## Making the Confusion Matrix

# In[33]:


# Making the Confusion Matrix according to Random Forest Classification to the Training set 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[34]:


## Making the Confusion Matrix according to SVM mode Classification to the Training set 
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Predicting if a single review is positive or negative
# ## Positive review
# Use our model to predict if the following review:
# 
# "I love this place so much"
# 
# is positive or negative.

# In[35]:


new_review = 'I love this place so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)


# The review was correctly predicted as positive by our model.

# ## Negative review
# Use our model to predict if the following review:
# 
# "The food is not that good."
# 
# is positive or negative.

# In[36]:


new_review = 'The food is not that good'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)


# 
# The review was correctly predicted as negative by our model.

# In[ ]:





# In[ ]:




