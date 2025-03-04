# %% Import packages
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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# %%
def split_dataframes(df_emails_preprocessed):
    """Splits the preprocessed email dataframe into two separate dataframes for the topics"""
    df_emails_T303 = df_emails_preprocessed[(df_emails_preprocessed["Topic 303"]==0) | (df_emails_preprocessed["Topic 303"]==1)]
    df_emails_T304 = df_emails_preprocessed[(df_emails_preprocessed["Topic 304"]==0) | (df_emails_preprocessed["Topic 304"]==1)]

    return df_emails_T303, df_emails_T304

df_emails_new = pd.read_csv('df_emails_new', index_col=0)
df_emails_new = df_emails_new[df_emails_new["Pre-processed body"].notna()]
df_emails_new["Tokens"] = df_emails_new["Tokens"].apply(ast.literal_eval)

df_emails_T303_new, df_emails_T304_new = split_dataframes(df_emails_new)

# %% Vectorize the email data
# tfidf
tfidfvect = TfidfVectorizer()
T303_tfidf_new = tfidfvect.fit_transform(df_emails_T303_new["Pre-processed body"]) 
T304_tfidf_new = tfidfvect.fit_transform(df_emails_T304_new["Pre-processed body"]) 

# word2vec
# Load the pre-trained word2vec model
word2vec_model = KeyedVectors.load("word2vec_pretrained_model")

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

T303_w2v_new = get_average_word2vec(df_emails_T303_new["Pre-processed body"])
T304_w2v_new = get_average_word2vec(df_emails_T304_new["Pre-processed body"])

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

T303_bert_new = get_BERT_embed(df_emails_T303_new["Pre-processed body"])
T304_bert_new = get_BERT_embed(df_emails_T304_new["Pre-processed body"])


# %% Tuning
def tuning_grid(data, classifier, grid, topic):
    """Performs hyperparameter tuning using 5-fold grid search CV 
    data: np.appray of the vectorized predictors of the email data of topic 'topic' 
    classifier: the sklearn classifier function of which the hyperparameters need to be tuned 
    grid: the grid that needs to be searched for the optimal set of hyperparameters
    topic: the topic of the email dataset that is used ('303'or '304')
    """
    if topic == '303':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, df_emails_T303_new["Topic 303"], test_size=0.33, random_state=20252001)
    elif topic == '304':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, df_emails_T304_new["Topic 304"], test_size=0.33, random_state=20252001)

    clf = classifier

    clf_gridsearch = GridSearchCV(estimator=clf, param_grid=grid, cv=5, n_jobs=-1, scoring='f1')

    clf_gridsearch.fit(X_train, y_train)

    y_pred = clf_gridsearch.predict(X_test)

    scores = [sklearn.metrics.f1_score(y_test, y_pred), sklearn.metrics.accuracy_score(y_test, y_pred),
              sklearn.metrics.recall_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred)]

    return clf_gridsearch, clf_gridsearch.best_params_, clf_gridsearch.best_score_, scores


def tuning_random(data, classifier, grid, topic, n_iter):
    """Performs hyperparameter tuning using 5-fold randomzied search CV
    data: np.appray of the vectorized predictors of the email data of topic 'topic' 
    classifier: the sklearn classifier function of which the hyperparameters need to be tuned 
    grid: the grid that needs to be searched for the optimal set of hyperparameters
    topic: the topic of the email dataset that is used ('303'or '304')
    n_iter: the number of hyperparameter settings to evaluate
    """
    if topic == '303':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, df_emails_T303_new["Topic 303"], test_size=0.33, random_state=20252001)
    elif topic == '304':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, df_emails_T304_new["Topic 304"], test_size=0.33, random_state=20252001)

    clf = classifier

    clf_gridsearch = RandomizedSearchCV(estimator=clf, param_distributions=grid, cv=5, n_jobs=-1, scoring='f1', n_iter=n_iter, random_state=20250124)

    clf_gridsearch.fit(X_train, y_train)

    y_pred = clf_gridsearch.predict(X_test)

    scores = [sklearn.metrics.f1_score(y_test, y_pred), sklearn.metrics.accuracy_score(y_test, y_pred),
              sklearn.metrics.recall_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred)]

    return clf_gridsearch, clf_gridsearch.best_params_, clf_gridsearch.best_score_, scores


# ---------------------------------------------------------------------


# Random Forest
rf_grid = {'n_estimators': [50, 100, 200, 400, 600, 800, 1000], 'min_samples_leaf': [1, 5, 10, 15, 20], 'max_features': [0.1, 0.2, 0.25, 0.3, 0.35]}
# Topic 303
rf_grid_results_303_bert = tuning_grid(T303_bert_new, RandomForestClassifier(), rf_grid, '303')
rf_grid_results_303_w2v = tuning_grid(T303_w2v_new, RandomForestClassifier(), rf_grid, '303')
rf_grid_results_303_tfidf = tuning_grid(T303_tfidf_new, RandomForestClassifier(), rf_grid, '303')
# Topic 304
rf_grid_results_304_bert = tuning_grid(T304_bert_new, RandomForestClassifier(), rf_grid, '304')
rf_grid_results_304_w2v = tuning_grid(T304_w2v_new, RandomForestClassifier(), rf_grid, '304')
rf_grid_results_304_tfidf = tuning_grid(T304_tfidf_new, RandomForestClassifier(), rf_grid, '304')


# ---------------------------------------------------------------------

# SVM
svm_grid_rbf = {'C': [1, 10, 15, 30, 50, 70, 100], 'gamma': [0.01, 0.1, 1, 10]}
# Topic 303
svm_grid_results_303_bert = tuning_grid(T303_bert_new, SVC(kernel='rbf'), svm_grid_rbf, '303')
svm_grid_results_303_w2v = tuning_grid(T303_w2v_new, SVC(kernel='rbf'), svm_grid_rbf, '303')
svm_grid_results_303_tfidf = tuning_grid(T303_tfidf_new, SVC(kernel='rbf'), svm_grid_rbf, '303')
# Topic 304
svm_grid_results_304_bert = tuning_grid(T304_bert_new, SVC(kernel='rbf'), svm_grid_rbf, '304')
svm_grid_results_304_w2v = tuning_grid(T304_w2v_new, SVC(kernel='rbf'), svm_grid_rbf, '304')
svm_grid_results_304_tfidf = tuning_grid(T304_tfidf_new, SVC(kernel='rbf'), svm_grid_rbf, '304')


# ----------------------------------------------------------------------

# Neural Network
nn_grid_adam = {"hidden_layer_sizes": [(50,), (100,), (150,), (50,50), (100,50), (100,100)], 
           "alpha": [0.0001, 0.001, 0.01, 0.1], "batch_size": [32, 64, 128], "learning_rate_init": [0.001, 0.01, 0.1]}


nn_grid_sgd = {"hidden_layer_sizes": [(50,), (100,), (150,), (50,50), (100,50), (100,100), (150, 150)], 
           "alpha": [0.0001, 0.001, 0.01, 0.1], "batch_size": [32, 64, 128], "learning_rate_init": [0.001, 0.01, 0.1], 
           "learning_rate": ['constant', 'adaptive']}


# Topic 303
nn_grid_results_303_bert_adam = tuning_grid(T303_bert_new, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam, '303')
nn_grid_results_303_w2v_adam = tuning_grid(T303_w2v_new, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam, '303')
# Topic 304
nn_grid_results_304_bert_adam = tuning_grid(T304_bert_new, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam, '304')
nn_grid_results_304_w2v_adam = tuning_grid(T304_w2v_new, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam, '304')

# Topic 303
nn_grid_results_303_bert_sgd = tuning_grid(T303_bert_new, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd, '303')
nn_grid_results_303_w2v_sgd = tuning_grid(T303_w2v_new, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd, '303')
# Topic 304
nn_grid_results_304_bert_sgd = tuning_grid(T304_bert_new, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd, '304')
nn_grid_results_304_w2v_sgd = tuning_grid(T304_w2v_new, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd, '304')


# TF-IDF (Randomized search is implemented here. The large number of hyperparameters in combination with the high-dimensionality of the data would otherwise make it too computationally expensive)
nn_grid_results_303_tfidf_adam = tuning_random(T303_tfidf_new, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam, '303', n_iter=20)
nn_grid_results_304_tfidf_adam = tuning_random(T304_tfidf_new, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam, '304', n_iter=20)

nn_grid_results_303_tfidf_sgd = tuning_random(T303_tfidf_new, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd, '303', n_iter=20)
nn_grid_results_304_tfidf_sgd = tuning_random(T304_tfidf_new, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd, '304', n_iter=20)


