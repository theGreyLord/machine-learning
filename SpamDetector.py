
'''
Solution
'''
import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Output printing out first 5 columns
df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
print(df.head()) # returns (rows, columns)

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words='english')

# documents = ['Hello, how are you!',
#                 'Win money, win from home.',
#                 'Call me now.',
#                 'Hello, Call hello you tomorrow?']
#
# count_vector.fit(documents)
# print(count_vector.get_feature_names())
#
# doc_array = count_vector.transform(documents).toarray()
#print(doc_array)

# frequency_matrix = pd.DataFrame(doc_array,
#                                 columns = count_vector.get_feature_names())
# print(frequency_matrix)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

#########################
## Diabetes Calculator ##
#########################

# # P(D)
# p_diabetes = 0.01
#
# # P(~D)
# p_no_diabetes = 0.99
#
# # Sensitivity or P(Pos|D)
# p_pos_diabetes = 0.9
#
# # Specificity or P(Neg/~D)
# p_neg_no_diabetes = 0.9
#
# # P(Pos)
# p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1 - p_neg_no_diabetes))
# print('The probability of getting a positive test result P(Pos) is: {}',format(p_pos))
#
# # P(D|Pos)
# p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos
# print('Probability of an individual having diabetes, given that that individual got a positive test result is:\
# ',format(p_diabetes_pos))
#
# # P(Pos/~D)
# p_pos_no_diabetes = 0.1
#
# # P(~D|Pos)
# p_no_diabetes_pos = (p_no_diabetes * p_pos_no_diabetes) / p_pos
# print 'Probability of an individual not having diabetes, given that that individual got a positive test result is:'\
# ,p_no_diabetes_pos

#########################
## Diabetes Calculator ##
#########################
