# %%
import shap
import sklearn.model_selection
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from sklearn.svm import SVC
import pandas as pd
import sklearn
from lime.lime_text import LimeTextExplainer
import numpy as np
from gensim.models import KeyedVectors
import ast
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed


# %%
def split_dataframes(df_emails_preprocessed):
    """Splits the preprocessed email dataframe into two separate dataframes for the topics"""
    df_emails_T303 = df_emails_preprocessed[(df_emails_preprocessed["Topic 303"]==0) | (df_emails_preprocessed["Topic 303"]==1)]
    df_emails_T304 = df_emails_preprocessed[(df_emails_preprocessed["Topic 304"]==0) | (df_emails_preprocessed["Topic 304"]==1)]

    return df_emails_T303, df_emails_T304

df_emails_new = pd.read_csv('df_emails_new', index_col=0)
df_emails_new = df_emails_new[df_emails_new["Pre-processed body"].notna()]

# delete emails that contain only 1 word
df_emails_new = df_emails_new[df_emails_new["Pre-processed body"].str.split().apply(len) > 1]

df_emails_new["Tokens"] = df_emails_new["Tokens"].apply(ast.literal_eval)

df_emails_T303, df_emails_T304 = split_dataframes(df_emails_new)

X_train_303, X_test_303, y_train_303, y_test_303 = sklearn.model_selection.train_test_split(df_emails_T303["Pre-processed body"], df_emails_T303["Topic 303"], test_size=0.3, random_state=20252001)
X_train_304, X_test_304, y_train_304, y_test_304 = sklearn.model_selection.train_test_split(df_emails_T304["Pre-processed body"], df_emails_T304["Topic 304"], test_size=0.3, random_state=20252001)

# %% Vectorize the email data
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

X_train_303_w2v = get_average_word2vec(X_train_303)
X_train_304_w2v = get_average_word2vec(X_train_304)


def get_average_word2vec_single(text, k=300):
    """Produces the word2vec k-dimensional embedding for a single text"""
    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    if len(text_tokens) < 1:
        text_embedding = [np.zeros(k)]
    else:
        text_embedding = [word2vec_model[token] if token in word2vec_model else np.zeros(k) for token in text_tokens]

    averaged_embedding = np.mean(text_embedding, axis=0)

    return averaged_embedding


# %% Training each of the classifiers using the hyperparameter settings found in '2. Tuning.py'
def smote_resample(X_train, y_train): 
  """Performs SMOTE sampling on the training data"""
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train

X_train_303_w2v, y_train_303 = smote_resample(X_train_303_w2v, y_train_303)
X_train_304_w2v, y_train_304 = smote_resample(X_train_304_w2v, y_train_304)

# rf
rf_T303_w2v = RandomForestClassifier(max_features= 0.35, min_samples_leaf= 1, n_estimators= 200)
rf_T304_w2v = RandomForestClassifier(max_features= 0.3, min_samples_leaf= 1, n_estimators= 100)

rf_T303_w2v.fit(X_train_303_w2v, y_train_303)
rf_T304_w2v.fit(X_train_304_w2v, y_train_304)

# svm
svm_T303_w2v = SVC(C= 10, gamma= 1, probability=True)
svm_T304_w2v = SVC(C= 30, gamma= 1, probability=True)

svm_T303_w2v.fit(X_train_303_w2v, y_train_303)
svm_T304_w2v.fit(X_train_304_w2v, y_train_304)

# nn
nn_T303_w2v = MLPClassifier(solver='adam', activation='relu', alpha=0.1, batch_size=128, hidden_layer_sizes=(50,), learning_rate_init=0.01)
nn_T304_w2v = MLPClassifier(solver='sgd', activation='relu', alpha=0.01, batch_size=64, hidden_layer_sizes=(100,), learning_rate='adaptive', learning_rate_init=0.1)

nn_T303_w2v.fit(X_train_303_w2v, y_train_303)
nn_T304_w2v.fit(X_train_304_w2v, y_train_304)

# %% Producing LIME explanations
def pipeline(x, classifier):
  """""Pipeline of text vectorization followed by calculating the prediction probability of an observation""" 
    # Transform the input text to TF-IDF vectors
    x_word2vec = get_average_word2vec_single(x)
    # Get the prediction scores from the Random Forest model
    return classifier.predict_proba(x_word2vec.reshape(1,-1))  


def xai_lime(X_test, index, classifier):
  """"Produces the LIME explanations for observation at index 'index' in the test set using classifier 'classifier'"""" 

    def pipeline(x):
        x_word2vec = get_average_word2vec(x)
        return classifier.predict_proba(x_word2vec)  # This will return a probability for each class

    sample_data = X_test.iloc[index]
    explainer = LimeTextExplainer(class_names=["irrelevant","relevant"])
    exp = explainer.explain_instance(sample_data, pipeline, num_features=50, labels=(0,1))

    return exp


def xai_lime_visual(X_test, index, classifier, n_features=15):
    """Special lime function that produces a LIME explanation with only the first n_features for a nicer visualization"""

    def pipeline(x):
        x_word2vec = get_average_word2vec(x)
        return classifier.predict_proba(x_word2vec)  # This will return a probability for each class

    sample_data = X_test.iloc[index]
    explainer = LimeTextExplainer(class_names=["irrelevant","relevant"])
    exp = explainer.explain_instance(sample_data, pipeline, num_features=n_features, labels=(0,1))

    return exp

def visualize_one_exp(exp, index, labels, class_names = ["irrelevant","relevant"]):
    """Produces a visualization for the LIME explanations of one observation at index 'index' in the test set"""
    print(f'Index: {index}')
    print(f'True class: {class_names[int(labels.iloc[index])]}')
    exp.show_in_notebook(text=True, labels=(1,))


# %% Producing SHAP explanations
def xai_shap(X_test, index, classifier):
    """Computes shap values for the email in the test set with index 'index' on the classifcation with classifier 'classifier'
    """
    def f(x):
        x_vectorized = get_average_word2vec(x)
        outputs = classifier.predict_proba(x_vectorized)  # This will return a probability for each class
        return outputs

    masker = shap.maskers.Text(r"\W")

    explainer = shap.Explainer(f, masker, output_names=["irrelevant", "relevant"])

    sample_texts = X_test[index:index+1]

    shap_values = explainer(sample_texts)

    return shap_values

def visualize_shap(exp, index, y_test, X_test, classifier, class_names=["irrelevant", "relevant"], n_features=15):
    """Produces a visualization of the SHAP explanations for one obsevations at index 'index' in the test set"""        
    predict_prob = pipeline(X_test.iloc[index], classifier)
    print(f'Index: {index}')
    print(f'True class: {class_names[int(y_test.iloc[index])]}')
    print(f'\nPrediction probabilities\nirrelevant: {predict_prob[0][0]}\nrelevant: {predict_prob[0][1]}')
    shap.plots.waterfall(exp[0,:,1], max_display=n_features)      


# %% Evaluation metrics: Fidelity
def Fidelity(X_test, classifier, xai, X_train, y_train, X_train_vec, n=200, n_feat = 5):
    """Computes the fidelity score of XAI method 'xai' for a particular classification model 'classifier'
    n: number of data points from the test set to compute the Fidelity for
    n_feat: number of feature to mask
    """

    def fidelity_one(X_test, index, classifier, xai):
        """Computes the fidelity of the explanation for one instance at index 'index'"""

        X_test_copy = X_test.copy(deep=True)

        def predict(X_test, index, classifier):
            """Vectorizes and makes classification prediction"""
            x = X_test.iloc[index]
            x_vectorized = get_average_word2vec_single(x).reshape(1,-1)

            # Determine the original prediction probabiltiy for the 1 class
            pred_prob = classifier.predict_proba(x_vectorized) # index 1 to select the probability of 1 class

            return pred_prob
        
        
        old_pred_prob = predict(X_test, index, classifier)
        old_pred_class = old_pred_prob.argmax()
        old_pred_prob_comp = predict(X_test, index, rf_comparing)[0][old_pred_class]

        def select_relevant_features(xai_words, n_feat=n_feat):
            """Selects the n_feat most relevant words as determined by the xai method"""
            relevant_features = []
            for word in xai_words:
                if word.strip() not in relevant_features:
                    relevant_features.append(word.strip())
                if len(relevant_features) == n_feat:
                    break
            return relevant_features
        
        # Run explainer model
        if xai == "shap":
            shap_values = xai_shap(X_test, index, classifier)
            # collect the x most influential features
            idx_positive_cont = np.where(shap_values[0,:,old_pred_class].values > 0)[0]
            ft_order = np.argsort(abs(shap_values[0,:,old_pred_class].values))[::-1]
            idx_pos_contr_order = np.array([f for f in ft_order if f in idx_positive_cont]).astype(int)
            ft_in_order_pos_contr = shap_values[0,:,old_pred_class].data[idx_pos_contr_order]
            top_features = select_relevant_features(ft_in_order_pos_contr) 
        elif xai == "lime":
            lime_values = xai_lime(X_test, index, classifier)
            lime_features = np.array(lime_values.as_list(label=old_pred_class))[:,0] # already in order from most to least influential (abs) so no need for reordering here
            # collect the x most influential features
            idx_positive_contr = np.where(np.array(lime_values.as_map()[old_pred_class])[:,1] > 0)
            lime_features_pred_class = lime_features[idx_positive_contr]
            top_features = select_relevant_features(lime_features_pred_class)

        def remove_features(text, top_features):
            """Remove the words in top_features fromthe text in 'text'"""
            return ' '. join([word for word in text.split() if word not in top_features])
        
        X_test_copy.iloc[index] = remove_features(X_test.iloc[index], top_features)

        new_pred_prob_comp = predict(X_test_copy, index, rf_comparing)[0][old_pred_class]

        mape = abs(new_pred_prob_comp - old_pred_prob_comp)/old_pred_prob_comp
            
        return mape
    
    # the untuned Random Forest classifier is constructed to calculate changes in prediction probabilities
    rf_comparing = RandomForestClassifier()
    rf_comparing.fit(X_train_vec, y_train)

    fidelity_list = []
    # Draw n random numbers from the 0 to len(X_test)
    indices = np.random.choice(range(0,len(X_test)), size=n, replace=False)

    for ind in indices:
        fidelity_list.append(fidelity_one(X_test, ind, classifier, xai)) 

    # Average over the fidelities
    fidelity_avg = np.array(fidelity_list).mean()

    return fidelity_avg


def Fidelity_sim(classifier, topic, xai, n_feat_list = [1,2,5,10]):
    """Computes the Fidelities of the 'xai' considering the 1,2,5 and 10 most relevant features using classifier 'classifier' on the email data of topic 'topic'"""
    fidel_list = np.zeros((len(n_feat_list), 5)) 
    for i in range(len(n_feat_list)):
        for j in range(5):
            if topic == '303':
                fidel = Fidelity(X_test_303, classifier, xai, X_train_303, y_train_303, X_train_303_w2v, 200, n_feat_list[i])
            elif topic == '304':
                fidel = Fidelity(X_test_304, classifier, xai, X_train_304, y_train_304, X_train_304_w2v, 200, n_feat_list[i])
            fidel_list[i,j] = fidel

            print(i,j)

    return fidel_list
        

def Fidelity_sim_parallel(classifier, topic, xai, n_feat_list = [1,2,5,10]):
    """Utilizes prarallel computing to compute the Fidelities"""
    def compute_fidelity(classifier, topic, xai, n_feat):
        if topic == '303':
            return Fidelity(X_test_303, classifier, xai, X_train_303, y_train_303, X_train_303_w2v, 200, n_feat)
        elif topic == '304':
            return Fidelity(X_test_304, classifier, xai, X_train_304, y_train_304, X_train_304_w2v, 200, n_feat)

    results = Parallel(n_jobs=-1)(delayed(compute_fidelity)(classifier, topic, xai, n_feat) for n_feat in n_feat_list for _ in range(5))

    fidel_list = np.array(results).reshape((len(n_feat_list), 5))

    return fidel_list

fidel_rf_303_lime = Fidelity_sim_parallel(rf_T303_w2v, '303', 'lime')
fidel_svm_303_lime = Fidelity_sim_parallel(svm_T303_w2v, '303', 'lime')
fidel_nn_303_lime = Fidelity_sim_parallel(nn_T303_w2v, '303', 'lime')
fidel_rf_304_lime = Fidelity_sim_parallel(rf_T304_w2v, '304', 'lime')
fidel_svm_304_lime = Fidelity_sim_parallel(svm_T304_w2v, '304', 'lime')
fidel_nn_304_lime = Fidelity_sim_parallel(nn_T304_w2v, '304', 'lime')

fidel_rf_303_shap = Fidelity_sim(rf_T303_w2v, '303', 'shap')
fidel_svm_303_shap = Fidelity_sim(svm_T303_w2v, '303', 'shap')
fidel_nn_303_shap = Fidelity_sim(nn_T303_w2v, '303', 'shap')
fidel_rf_304_shap = Fidelity_sim(rf_T304_w2v, '304', 'shap')
fidel_svm_304_shap = Fidelity_sim(svm_T304_w2v, '304', 'shap')
fidel_nn_304_shap = Fidelity_sim(nn_T304_w2v, '304', 'shap')


# %% Evaluation metrics: Stability
def Stability(X_test, classifier, n_top = 5, m=20, n=200):
  """Computes the Stability of the explanations of LIME in the test set using classifier 'classifier'
  n_top: the number of most relevant features to consider
  m: the number of iterations to run the xai method on the same observations
  n: number of datapoints in the test set to compute the Stability for
  """
    def explain_one_instance(X_test, index, classifier):
        """Produce the LIME explanations for one observations from the test set at index 'index' using classifier 'classifier'"""
        def select_relevant_features_attributions(ft, attr, n_feat=n_top): # TODO check, bc it doesn't do any ordering, so might not pick the x most relevant features
            relevant_features = []
            feature_attributions = []
            for i in range(len(ft)):
                if ft[i].strip() not in relevant_features:
                    relevant_features.append(ft[i].strip())
                    feature_attributions.append(attr[i])
                if len(relevant_features) == n_feat:
                    break
            return relevant_features, feature_attributions
        
        # Now run explainer model
        lime_values = xai_lime(X_test, index, classifier)
        features = np.array(lime_values.as_list(label=1))[:,0]
        attributions = np.array(lime_values.as_list(label=1))[:,1].astype(float) 
        # collect the x most influential features
        top_features, feature_attributions = select_relevant_features_attributions(features, attributions)

        return top_features, feature_attributions
    
    def VSI(ft0, ft1):
        """Computes the VSI of two explanations"""
        return len(set(ft0).intersection(set(ft1)))/len(set(ft0).union(set(ft1)))
    
    def CSI(attr0, attr1, ft0, ft1):
        """Computes the CSI of two explanations"""
        ft_set = set(ft0+ft1)
        attr_diff_list = []
        for word in ft_set:
            if (word in ft0) and (word in ft1):
                attr_diff = 1 - abs(attr0[ft0.index(word)] - attr1[ft1.index(word)])/abs(attr0[ft0.index(word)])
            elif (word in ft0) and (word not in ft1):
                attr_diff = 0 
            elif (word not in ft0) and (word in ft1):
                attr_diff = 0 
            attr_diff_list.append(attr_diff)
        
        return np.array(attr_diff_list).mean()


    indices = np.random.choice(range(0,len(X_test)), size=n, replace=False) # randomly choose indices from the test set
    VSI_ult_list = []
    CSI_ult_list = []

    for ind in indices:
        ft_list = []
        attr_list = []
        for it in range(m):
            ft, attr = explain_one_instance(X_test, ind, classifier)
            ft_list.append(ft)
            attr_list.append(attr)
    
        VSI_list = []
        CSI_list = []    
        for i in range(m):
            for j in range(i+1, m):
                VSI_list.append(VSI(ft_list[i], ft_list[j]))
                CSI_list.append(CSI(attr_list[i], attr_list[j], ft_list[i], ft_list[j]))
            
        VSI_ult_list.append(np.array(VSI_list).mean())
        CSI_ult_list.append(np.array(CSI_list).mean())

    
    return np.array(VSI_ult_list).mean(), np.array(CSI_ult_list).mean()


def Stability_sim(classifier, topic, n_top_list = [1,3,5,10]):
    """Computes the Stabilities of the LIME considering the 1,3,5 and 10 most relevant features using classifier 'classifier' on the email data of topic 'topic'"""
    stabil_list = np.zeros((len(n_top_list), 5))
    for i in range(len(n_top_list)):
        for j in range(5):
            if topic == '303':
                stabil = Stability(X_test_303, classifier, n_top_list[i], 20, 200 )
            elif topic == '304':
                stabil = Stability(X_test_304, classifier, n_top_list[i], 20, 200 )
            stabil_list[i,j] = stabil

            print(i,j)

    return stabil_list
        

def Stability_sim_par(classifier, topic, n_top_list = [1,3,5,10]):
    """Utilizes prarallel computing to compute the Stabilities"""
    def compute_stability(classifier, topic, xai, n_top):
        if topic == '303':
            return Stability(X_test_303, classifier, n_top, 10, 200)
        elif topic == '304':
            return Stability(X_test_304, classifier, n_top, 10, 200)

    results = Parallel(n_jobs=-1)(delayed(compute_stability)(classifier, topic, xai, n_top) for n_top in n_top_list for _ in range(5))

    stabil_list = np.array(results).reshape((len(n_top_list), 5))

    return stabil_list

stabil_rf_303_lime = Stability_sim_par(rf_T303_w2v, '303')
stabil_svm_303_lime = Stability_sim_par(svm_T303_w2v, '303')
stabil_nn_303_lime = Stability_sim_par(nn_T303_w2v, '303')
stabil_rf_304_lime = Stability_sim_par(rf_T304_w2v, '304')
stabil_svm_304_lime = Stability_sim_par(svm_T304_w2v, '304')
stabil_nn_304_lime = Stability_sim_par(nn_T304_w2v, '304')

