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

# ALSO DO BELOW TWO LINES AT OTHER PLACES WHEN THE DATA IS LOADED. WITHOUT THIS, THE TOKENS ARE STRINGS ('[...]')
df_emails_new["Tokens"] = df_emails_new["Tokens"].apply(ast.literal_eval)

df_emails_T303, df_emails_T304 = split_dataframes(df_emails_new)



# %%

# ######################################################################################
# --------------------- Train and run the models ---------------------
# ######################################################################################

def classification(classifier, data, topic):
    """Trains and runs the ML models to classify the emails
    
    classifier: mlp, randomforestclassifier or SVC with hyperparameters
    data: vectorized email data (so after running one of the vectorization functions)

    output: 
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
    model = sklearn.neural_network.MLPClassifier()

    scores = [sklearn.metrics.f1_score(y_test, y_pred), sklearn.metrics.accuracy_score(y_test, y_pred),
              sklearn.metrics.recall_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred)]


    return model, y_pred, scores

# %%
# tfidf
tfidfvect = TfidfVectorizer()
T303_tfidf = tfidfvect.fit_transform(df_emails_T303["Pre-processed body"]) 
T304_tfidf = tfidfvect.fit_transform(df_emails_T304["Pre-processed body"]) 

# word2vec
word2vec_model = KeyedVectors.load("word2vec_pretrained_model")

def get_average_word2vec(text_list, k=300):
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("BERT_tokenizer")
model = AutoModel.from_pretrained("BERT_model").to(device)

def get_BERT_embed(text_list):
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
# rf
rf_T303_tfidf = classification(RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 200), T303_tfidf, '303')
#  [0.631578947368421,  0.8403755868544601,  0.6517571884984026,  0.6126126126126126])
rf_T304_tfidf = classification(RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 100), T304_tfidf, '304')
# [0.45714285714285713,  0.8201892744479495,  0.4332129963898917,  0.4838709677419355])


rf_T303_w2v = classification(RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 200), T303_w2v, '303')
#  [0.5897435897435898,  0.8497652582159625,  0.5143769968051118,  0.6909871244635193])
rf_T304_w2v = classification(RandomForestClassifier(max_features= 0.3, min_samples_leaf= 1, n_estimators= 100), T304_w2v, '304')
#  [0.4434782608695652,  0.8384858044164037,  0.36823104693140796,  0.5573770491803278])


# svm
svm_T303_tfidf = classification(SVC(C= 15, gamma= 0.1), T303_tfidf, '303')
# [0.6184873949579832,  0.8477531857813548,  0.5878594249201278,  0.6524822695035462])
svm_T304_tfidf = classification(SVC(C=50, gamma= 0.1), T304_tfidf, '304')
#  [0.49230769230769234,  0.8334384858044164,  0.4620938628158845,  0.5267489711934157])


svm_T303_w2v = classification(SVC(C= 10, gamma= 1), T303_w2v, '303')
#  [0.6607669616519174,  0.8457411133467472,  0.7156549520766773,  0.6136986301369863])
svm_T304_w2v = classification(SVC(C= 30, gamma= 1), T304_w2v, '304')
#  [0.5063291139240507,  0.8277602523659306,  0.5054151624548736,  0.5072463768115942])


# nn
nn_T303_tfidf = classification(MLPClassifier(solver='sgd', activation='relu', alpha=0.001, batch_size=128, hidden_layer_sizes=(150,), learning_rate_init=0.01, learning_rate='adaptive'), T303_tfidf, '303')
# [0.6341463414634146,  0.8490945674044266,  0.6230031948881789,  0.6456953642384106])
nn_T304_tfidf = classification(MLPClassifier(solver='sgd', activation='relu', alpha=0.001, batch_size=128, hidden_layer_sizes=(150,), learning_rate_init=0.01, learning_rate='adaptive'), T304_tfidf, '304')
#  [0.5088967971530249,  0.8258675078864354,  0.516245487364621,  0.5017543859649123])


nn_T303_w2v = classification(MLPClassifier(solver='adam', activation='relu', alpha=0.1, batch_size=128, hidden_layer_sizes=(50,), learning_rate_init=0.01), T303_w2v, '303')
#  [0.6609195402298851,  0.8417169684775319,  0.7348242811501597,  0.6005221932114883]) 
nn_T304_w2v = classification(MLPClassifier(solver='sgd', activation='relu', alpha=0.01, batch_size=64, hidden_layer_sizes=(100,), learning_rate='adaptive', learning_rate_init=0.1), T304_w2v, '304')
#  [0.5161290322580645,  0.8296529968454258,  0.51985559566787,  0.5124555160142349])


#### BERT
rf_T303_bert = classification(RandomForestClassifier(max_features= 0.25, min_samples_leaf= 5, n_estimators= 100), T303_bert, '303')
#  [0.6182432432432432,  0.8484238765928906,  0.5846645367412141,  0.6559139784946236])
svm_T303_bert = classification(SVC(C=50, gamma=0.01), T303_bert, '303')
#  [0.605475040257649,  0.8356807511737089,  0.6006389776357828,  0.6103896103896104])
nn_T303_bert = classification(MLPClassifier(activation='relu', solver='sgd', alpha=0.001, batch_size=128, hidden_layer_sizes=(50,), learning_rate_init=0.1, learning_rate='constant'), T303_bert, '303')
#  [0.5898305084745763,  0.8376928236083165,  0.5559105431309904,  0.628158844765343])


rf_T304_bert = classification(RandomForestClassifier(max_features= 0.3, min_samples_leaf= 1, n_estimators= 400), T304_bert, '304')
# [0.4197002141327623,  0.8290220820189275,  0.35379061371841153,  0.5157894736842106])
svm_T304_bert = classification(SVC(C=100, gamma=0.01), T304_bert, '304')
# [0.47957371225577267,  0.8151419558359622,  0.48736462093862815,  0.47202797202797203])
nn_T304_bert = classification(MLPClassifier(activation='relu', solver='sgd', alpha=0.1, batch_size=32, hidden_layer_sizes=(50,), learning_rate_init=0.01, learning_rate='constant'), T304_bert, '304')
#  [0.5076923076923077,  0.7577287066246057,  0.7148014440433214,  0.39363817097415504])


# %%
