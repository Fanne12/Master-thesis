# %% Import packages
import shap
import sklearn.cluster
import sklearn.decomposition
import sklearn.ensemble
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from sklearn.svm import SVC
import pandas as pd
import sklearn
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import ast
import sklearn
import sklearn.neural_network
from imblearn.over_sampling import SMOTE

# %%
def split_dataframes(df_emails_preprocessed):
    """Splits the preprocessed email dataframe into two separate dataframes for the topics"""
    df_emails_T303 = df_emails_preprocessed[(df_emails_preprocessed["Topic 303"]==0) | (df_emails_preprocessed["Topic 303"]==1)]
    df_emails_T304 = df_emails_preprocessed[(df_emails_preprocessed["Topic 304"]==0) | (df_emails_preprocessed["Topic 304"]==1)]

    return df_emails_T303, df_emails_T304

df_emails_new = pd.read_csv('df_emails_new', index_col=0)
df_emails_new = df_emails_new[df_emails_new["Pre-processed body"].notna()]
df_emails_new["Tokens"] = df_emails_new["Tokens"].apply(ast.literal_eval)

df_emails_T303, df_emails_T304 = split_dataframes(df_emails_new)

# %% Vectorize the email data
# tfidf
tfidfvect = TfidfVectorizer()
T303_tfidf = tfidfvect.fit_transform(df_emails_T303["Pre-processed body"]) 
T304_tfidf = tfidfvect.fit_transform(df_emails_T304["Pre-processed body"]) 

# word2vec
# Load the pre-trained word2vec model
word2vec_model = api.load("word2vec-google-news-300") 

def get_average_word2vec(text_list, k=300):
    """Produces word2vec k-dimensional embeddings for the text in text_list"""
    tokenizer = RegexpTokenizer(r'\w+')
    embedding_list = []
    for text in text_list:
        text_tokens = tokenizer.tokenize(text)
        if len(text_tokens) < 1:
            text_embedding = [np.zeros(k)]
        else:
            text_embedding = [word2vec_model[token] if token in word2vec_model else np.zeros(k) for token in text_tokens]

        averaged_embedding = np.mean(text_embedding, axis=0)
        embedding_list.append(averaged_embedding)

    return np.array(embedding_list)

T303_w2v = get_average_word2vec(df_emails_T303["Pre-processed body"])
T304_w2v = get_average_word2vec(df_emails_T304["Pre-processed body"])


# bert
# Load the pre-trained BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

def get_BERT_embed(text_list):
    """Produces BERT embeddings for the text in text_list"""
    embedding_list = []
    for text in text_list:
        texts_BERT = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            hidden_train = model(**texts_BERT)
        embedding_list.append(hidden_train.last_hidden_state[:,0,:][0])

    return np.array(embedding_list)

T303_bert = get_BERT_embed(df_emails_T303["Pre-processed body"])
T304_bert = get_BERT_embed(df_emails_T304["Pre-processed body"])

# %%
# ######################################################################################
# --------------------- Train and run the models ---------------------
# ######################################################################################

def classification(classifier, data, topic):
    """Trains and runs the ML models to classify the emails
    classifier: MLP(), RandomForestClassifier() or SVC() with hyperparameters resulting from tuning in '2. Tuning.py'
    data: vectorized email data of topic 'topic'
    topic: the topic of the vectorized email data ('303' or '304')
    output: [trained classifier, predictions, scoring metrics]
    """

    if topic == '303':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, df_emails_T303["Topic 303"], test_size=0.3, random_state=20252001)
    elif topic == '304':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, df_emails_T304["Topic 304"], test_size=0.3, random_state=20252001)


    # SMOTE sampling
    smote = SMOTE(sampling_strategy='minority') # TODO check if want to specify additional parameters
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    model = classifier

    # Train the model
    model.fit(X_train, y_train)

    # Classify test set
    y_pred = model.predict(X_test)

    scores = [sklearn.metrics.f1_score(y_test, y_pred), sklearn.metrics.accuracy_score(y_test, y_pred),
              sklearn.metrics.recall_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred)]


    return model, y_pred, scores


# %%
# rf
rf_T303_tfidf = classification(RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 200), T303_tfidf, '303')
rf_T304_tfidf = classification(RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 100), T304_tfidf, '304')

rf_T303_w2v = classification(RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 200), T303_w2v, '303')
rf_T304_w2v = classification(RandomForestClassifier(max_features= 0.3, min_samples_leaf= 1, n_estimators= 100), T304_w2v, '304')

rf_T303_bert = classification(RandomForestClassifier(max_features= 0.25, min_samples_leaf= 5, n_estimators= 100), T303_bert, '303')
rf_T304_bert = classification(RandomForestClassifier(max_features= 0.3, min_samples_leaf= 1, n_estimators= 400), T304_bert, '304')

# svm
svm_T303_tfidf = classification(SVC(C= 15, gamma= 0.1), T303_tfidf, '303')
svm_T304_tfidf = classification(SVC(C=50, gamma= 0.1), T304_tfidf, '304')

svm_T303_w2v = classification(SVC(C= 10, gamma= 1), T303_w2v, '303')
svm_T304_w2v = classification(SVC(C= 30, gamma= 1), T304_w2v, '304')

svm_T303_bert = classification(SVC(C=50, gamma=0.01), T303_bert, '303')
svm_T304_bert = classification(SVC(C=100, gamma=0.01), T304_bert, '304')

# nn
nn_T303_tfidf = classification(MLPClassifier(solver='sgd', activation='relu', alpha=0.001, batch_size=128, hidden_layer_sizes=(150,), learning_rate_init=0.01, learning_rate='adaptive'), T303_tfidf, '303')
nn_T304_tfidf = classification(MLPClassifier(solver='sgd', activation='relu', alpha=0.001, batch_size=128, hidden_layer_sizes=(150,), learning_rate_init=0.01, learning_rate='adaptive'), T304_tfidf, '304')

nn_T303_w2v = classification(MLPClassifier(solver='adam', activation='relu', alpha=0.1, batch_size=128, hidden_layer_sizes=(50,), learning_rate_init=0.01), T303_w2v, '303')
nn_T304_w2v = classification(MLPClassifier(solver='sgd', activation='relu', alpha=0.01, batch_size=64, hidden_layer_sizes=(100,), learning_rate='adaptive', learning_rate_init=0.1), T304_w2v, '304')

nn_T303_bert = classification(MLPClassifier(activation='relu', solver='sgd', alpha=0.001, batch_size=128, hidden_layer_sizes=(50,), learning_rate_init=0.1, learning_rate='constant'), T303_bert, '303')
nn_T304_bert = classification(MLPClassifier(activation='relu', solver='sgd', alpha=0.1, batch_size=32, hidden_layer_sizes=(50,), learning_rate_init=0.01, learning_rate='constant'), T304_bert, '304')

