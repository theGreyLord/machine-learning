'''
Solution:
'''
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

# lower_case_documents = []
# for i in documents:
#     lower_case_documents.append(i.lower())
# print(lower_case_documents)
#
#
# '''
# Solution:
# '''
# sans_punctuation_documents = []
# import string
#
# for i in lower_case_documents:
#     sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
# print(sans_punctuation_documents)


'''
Solution
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

#print(count_vector)

count_vector.fit(documents)
print(count_vector.get_feature_names())

doc_array = count_vector.transform(documents).toarray()
#print(doc_array)

frequency_matrix = pd.DataFrame(doc_array,
                                columns = count_vector.get_feature_names())
print(frequency_matrix)
