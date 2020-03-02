#Problem Statement : Accurately classifing spam messages by building a predictive model.

#Solution : Data preprocessing and predictive analysis using naive bayes


#1.Import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#2.Load the data
messages = pd.read_csv('C:/Users/kavya/Desktop/npl/smsspamcollection', sep='\t',
                           names=["label", "message"])

#3. Download stopwards
nltk.download('stopwords')

#4.Data Preprocessing
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#5. Creating the Bag of Words model

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

#6.Train and Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#7. Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

#8. Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

#9.Model Evaluation 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

accuracy

