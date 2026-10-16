import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = ["free money win lottery", "hi how are you", "win free prize now", "let's meet tomorrow", "congratulations you won"]
labels = [1, 0, 1, 0, 1]

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data)

model = MultinomialNB()
model.fit(X, labels)

pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(model, open('model.pkl','wb'))

print("Ho gaya! 2 files ban gayi")