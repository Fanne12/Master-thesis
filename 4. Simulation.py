# %% Import packages
import lime.lime_tabular
import lime.lime_text
import shap
import sklearn.cluster
import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neighbors
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from sklearn.svm import SVC
import pandas as pd
import sklearn
import lime
import numpy as np
from gensim.models import KeyedVectors
import scipy as sp
import statistics
import matplotlib.pyplot as plt
import ast
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed


# %%
def split_dataframes(df_emails_preprocessed):
    """Splits the preprocessed email dataframe into two separate dataframes for the topics"""
    df_emails_T303 = df_emails_preprocessed[(df_emails_preprocessed["Topic 303"]==0) | (df_emails_preprocessed["Topic 303"]==1)]
    df_emails_T304 = df_emails_preprocessed[(df_emails_preprocessed["Topic 304"]==0) | (df_emails_preprocessed["Topic 304"]==1)]

    return df_emails_T303, df_emails_T304

df_emails = pd.read_csv('df_emails_new', index_col=0)
df_emails = df_emails[df_emails["Pre-processed body"].notna()]
df_emails["Tokens"] = df_emails["Tokens"].apply(ast.literal_eval)

df_emails_T303, df_emails_T304 = split_dataframes(df_emails)

# %% Vectorize the email data
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

T303_w2v = get_average_word2vec(df_emails_T303["Pre-processed body"])
T304_w2v = get_average_word2vec(df_emails_T304["Pre-processed body"])


# %% PCA 
n_PC = 10 # the number of principal components

def f_pca(data, n_PC, vectorization, topic):
    """ Runs a PCA analysis
    Data: the dataset containing specific topic and vectorization
    n_PC: number of principal components
    Output: [ratios of explained variance, loadings on the principal compoments]
    """
    # First: standardize the data
    if vectorization != 'tfidf':
        data_vectorized = StandardScaler().fit_transform(data)
    else:
        data_vectorized = data

    # Peform PCA
    pca = sklearn.decomposition.PCA(n_components=n_PC)
    data_pca = pca.fit_transform(data_vectorized)

    var_ratio = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T)

    # Make a scree plot of the explained variances
    plt.bar(range(1, n_PC+1), var_ratio)
    plt.plot(range(1, n_PC+1), np.cumsum(var_ratio), label = 'Cumulative Explained Variance Ratio')
    plt.title(f'Scree plot (Topic {topic}, {vectorization})')
    plt.xlabel('Number of principal components')
    plt.ylabel('Explained variance')
    plt.show()

    return var_ratio, loadings

pca_T303_w2v = f_pca(T303_w2v, n_PC, 'w2v', '303')
pca_T304_w2v = f_pca(T304_w2v, n_PC, 'w2v', '304')


# %% Distribution
# Kurtosis and skewness
def calc_kurtosis(data):
    """Calculates the kurtosis of each predictor"""
    return np.array([sp.stats.kurtosis(data[:,i]) for i in range(len(data[0]))])

def calc_skew(data):
    """Calculates the skewness of each predictor"""
    return np.array([sp.stats.skew(data[:,i]) for i in range(len(data[0]))])

kurt_w2v_303 = calc_kurtosis(T303_w2v) 
skew_w2v_303 = calc_skew(T303_w2v) 
plt.hist(kurt_w2v_303)
plt.hist(skew_w2v_303)

kurt_w2v_304 = calc_kurtosis(T304_w2v)
skew_w2v_304 = calc_skew(T304_w2v) 
plt.hist(kurt_w2v_304)
plt.hist(skew_w2v_304)

# Kilmogorov-Smirnov test
def test_ks(data):
  """Estimates the Maximum Likelihood parameter estimates of the student's t-distribution for each of the predictors in 'data' and computes the corresponding p-values for the Kilmogorov-Smirnov test
  Output: [p-values, degrees of freedom, locations, scales]
  """
    kstest_t_w2v = []
    df_lst = []
    loc_lst = []
    scale_lst = []
    for i in range(len(data[0])):
        v, loc, scale = sp.stats.t.fit(data[:,i])
        kstest_t_w2v.append(sp.stats.kstest(data[:,i], 't', [v, loc, scale])[1]) 
        df_lst.append(v)
        loc_lst.append(loc)
        scale_lst.append(scale)
        if i % 50 == 0:
            print(i)
    return np.array(kstest_t_w2v), np.array(df_lst), np.array(loc_lst), np.array(scale_lst)

kst_test_303, df_303, loc_303, scale_303 = test_ks(T303_w2v) 
plt.hist(df_303, bins=10) 
plt.hist(loc_303, bins=10) 
plt.hist(scale_303, bins=20) 

kst_test_304, df_304, loc_304, scale_304 = test_ks(T304_w2v) 
plt.hist(df_304, bins=10) 
plt.hist(loc_304, bins=10) 
plt.hist(scale_304, bins=20) 

# Plot the distribution of the data with student's t-distribution with Maximum Likelihood estimates
def plot_distr(data, index, nbins):
  """"Produces a figure with a histogram of the distribution of a predictor in the data together with the student's t-distribution with Maximum Likelihood estimates
  data: np.array predictor matrix
  index: column index of the predictor that you want to plot
  nbins: number of bins for the histogram 
  """ 
    x_axis = np.arange(-(abs(data[:,index])).max(), (abs(data[:,index])).max(), 0.01)
    v, loc, scale = sp.stats.t.fit(data[:,index])
    plt.hist(data[:,index], bins=nbins, density=True)
    plt.plot(x_axis, sp.stats.t.pdf(x_axis, v, loc, scale))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

plot_distr(T303_w2v, 0, 40)

plot_distr(T304_w2v, 0, 40)


# Pearon's correlation coefficients 
corr_w2v_303 = np.corrcoef(T303_w2v.T)
corr_w2v_304 = np.corrcoef(T304_w2v.T)

# Covariance matrix
cov_w2v_303 = np.cov(T303_w2v.T)
cov_w2v_304 = np.cov(T304_w2v.T)


# %% Overlap
def r_aug(X_data, y_data, k, theta):
    """Computes the augmented R-value to measure class overlap
    X_data: np.array predictor matrix
    y_data: vector containing the labels y 
    k: number of nearest neighbours to take into account 
    theta: threshold value """
    distance_matrix = sklearn.metrics.pairwise.euclidean_distances(X_data, X_data)

    class_neg = np.where(y_data==0)[0]
    class_pos = np.where(y_data==1)[0]
    IR = len(class_neg)/len(class_pos)

    r = np.zeros(2)
    for c in range(2): # class 0 and class 1
        if c==0:
            class_idx = class_neg
        elif c==1:
            class_idx = class_pos
        
        for j in class_idx:
            nn_j = np.argsort(distance_matrix[j])[1:k+1] # start from 1 to k+1 because every point is also its own nearest neighbour
            x = (y_data.iloc[nn_j]==1-c).sum() - theta
            if x > 0:
                r[c] += 1

        r[c] = r[c]/len(class_idx)

    r_augmented = (1/(1+IR)) * (r[0] + IR*r[1])

    return r_augmented

r_aug(T303_w2v, df_emails_T303["Topic 303"], 7, 3) # 0.35307469743095365
r_aug(T304_w2v, df_emails_T304["Topic 304"], 7, 3) # 0.3925972057379872

# Code used to create visualizations of varying degress of overlap (note, DGP_new_2 function
X, y, X_noise, cov_sim, b = DGP(5000, 2, 0, '303', 0, 0, 20, 0, 1) #40 (20.96), 16 (35.69), 0.5 (50.73)
print(r_aug(X, pd.DataFrame(y)[0], 7, 3))
plt.scatter(X[:,0], X[:,1], c=y, s=25)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# %% Function to construct the synthetic data
def construct_cov(n_feat_rel, n_feat_unrel, topic, r, seed):
  """"Construct the covariance matrix of the synthetic datasets, given the number of relevant features n_feat_rel, irrelevant features n_feat_unrel and desired correlation coefficient r"""
    np.random.seed(seed)

    cov_sim = np.zeros((n_feat_rel+n_feat_unrel, n_feat_rel+n_feat_unrel))

    if topic == '303':
        cov = np.cov(T303_w2v.T) 
    elif topic == '304':
        cov = np.cov(T304_w2v.T)

    # Construct new covariance matrix using cholesky decomposition with perturbations
    cholesky_factor = np.linalg.cholesky(cov)
    noise_level = 0.01

    perturbation = np.tril(np.random.normal(0, noise_level, size=cholesky_factor.shape))
    perturbed_cholesky_factor = cholesky_factor + perturbation

    new_sigma = np.dot(perturbed_cholesky_factor, perturbed_cholesky_factor.T)

    cov_sim_rel = cov_sim[:n_feat_rel, :n_feat_rel]
    cov_sim_unrel = cov_sim[n_feat_rel: , n_feat_rel:]

    cov_sim_rel[:,:] = new_sigma[:n_feat_rel, :n_feat_rel]
    cov_sim_unrel[:,:] = new_sigma[n_feat_rel:n_feat_rel+n_feat_unrel, n_feat_rel:n_feat_rel+n_feat_unrel]

    # Compute covariances between relevant and irrelevant variables based on the correlation coefficient r
    covariances = np.dot(np.sqrt(np.diag(cov_sim_rel)).reshape(-1,1), np.sqrt(np.diag(cov_sim_unrel)).reshape(1,-1)) * r
    
    covariances_diagonal = np.diag(covariances)
    np.fill_diagonal(cov_sim[:n_feat_rel, n_feat_rel:], covariances_diagonal)
    np.fill_diagonal(cov_sim[n_feat_rel:, :n_feat_rel], covariances_diagonal)

    # Check if covariance matrix is positive semi definite, if not, take closest psd to that matrix
    def nearest_psd(matrix):
        eigval, eigvec = np.linalg.eigh(matrix)
        eigval[eigval < 0] = 0
        return eigvec @ np.diag(eigval) @ np.linalg.inv(eigvec)

    cov_sim_psd = nearest_psd(cov_sim)

    return cov_sim_psd


def select_b(beta_range, n_feat_rel):
  """Draws n_feat_rel coefficients for the relevant features from the list of possible beta values beta_range"""
    betas = []
    for i in range(n_feat_rel // len(beta_range) + 1):
        if i == 0:
            betas.extend(random.sample(beta_range, n_feat_rel % len(beta_range)))
        else:
            betas.extend(random.sample(beta_range, len(beta_range)))
    return np.array(betas)


def construct_X(n_obs, n_feat_rel, n_feat_unrel, cov_sim, seed):  
    """Constructs the predictor matrix X, given the number of observations n_obs, the number of relevant n_feat_rel and irrelevant features n_feat_unrel and covariance matrix"""
    X = sp.stats.multivariate_t.rvs(loc=np.zeros(n_feat_rel+n_feat_unrel), shape=cov_sim, df=3, size=n_obs, random_state=seed)

    return X
  
def construct_y(X, n_obs, n_feat_rel, beta_range, b_factor, intercept, seed):
  """Given the predictor matrix X, number of observations n_obs, number of relevant features n_feat_rel, list of possible beta values beta_range, b_factor and intercept, constructs the response vector y
  Output: [response vector y, vector of coefficients beta]
  """
    b_range = list(b_factor*np.array(beta_range)) # to change the overlap

    random.seed(seed)
    b = select_b(b_range, n_feat_rel)

    expression = np.exp(intercept*np.ones(n_obs) + np.matmul(X[:, :n_feat_rel], b)) # select only the relevant features to enter the formula
    probability =  np.divide(expression, 1 + expression) 
    
    # to make sure that probabilites are always between 0 and 1
    probability[np.where(expression == np.inf)] = 1
    probability[np.where(expression == -np.inf)] = 0

    # using binomial distribution
    np.random.seed(seed)
    y = np.random.binomial(1, probability)

    return y, b

def DGP(n_obs, n_feat_rel, n_feat_unrel, topic, r, g_noise, b_factor, intercept, seed):
  """Collects the function above into one function to construct the covariance matrix, predictor matrix, coefficients vector and response vector. In the end, it also adds noise to the predictor matrix"""
    cov_sim = construct_cov(n_feat_rel, n_feat_unrel, topic, r, seed)
    X = construct_X(n_obs, n_feat_rel, n_feat_unrel, cov_sim, seed)
    beta_range = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
    y, b = construct_y(X, n_obs, n_feat_rel, beta_range, b_factor, intercept, seed)

    # Add gaussian noise to predictor space
    np.random.seed(seed)
    noise = np.random.normal(0, 1, X.shape) * X * g_noise
    X_noise = X + noise

    return X, y, X_noise, cov_sim, b


# %% Hyperparameter tuning grids and functions
rf_grid = {'n_estimators': [50, 100, 200, 400, 600, 800, 1000], 'min_samples_leaf': [1, 5, 10, 15, 20], 'max_features': [0.1, 0.2, 0.25, 0.3, 0.35]}

svm_grid_rbf = {'C': [1, 10, 15, 30, 50, 70, 100], 'gamma': [0.01, 0.1, 1, 10]}

nn_grid_adam = {"hidden_layer_sizes": [(50,), (100,), (150,), (50,50), (100,50), (100,100)], 
           "alpha": [0.0001, 0.001, 0.01, 0.1], "batch_size": [32, 64, 128], "learning_rate_init": [0.001, 0.01, 0.1]}

nn_grid_sgd = {"hidden_layer_sizes": [(50,), (100,), (150,), (50,50), (100,50), (100,100), (150, 150)], 
           "alpha": [0.0001, 0.001, 0.01, 0.1], "batch_size": [32, 64, 128], "learning_rate_init": [0.001, 0.01, 0.1], 
           "learning_rate": ['constant', 'adaptive']}


def tuning_grid_extra(X_data, y_data, classifier, grid):
    """Performs hyperparameter tuning using 5-fold gridsearch CV
    X_data: predictor matrix
    y_data: response vector
    classifier: the classifier of whicht the hyperparameters need to be tuned
    grid: the hyperparameter grid to search for the optimal settings """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, y_data, test_size = 0.33, random_state=0)

    clf = classifier

    clf_gridsearch = GridSearchCV(estimator=clf, param_grid=grid, cv=5, n_jobs=-1, scoring='f1')

    clf_gridsearch.fit(X_train, y_train)

    y_pred = clf_gridsearch.predict(X_test)

    scores = [sklearn.metrics.f1_score(y_test, y_pred), sklearn.metrics.accuracy_score(y_test, y_pred),
              sklearn.metrics.recall_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred)]

    return clf_gridsearch.best_params_, scores, clf_gridsearch.best_estimator_


def only_tuning(X_noise, y):
    rf_tuning = tuning_grid_extra(X_noise, y, RandomForestClassifier(), rf_grid)
    svm_tuning = tuning_grid_extra(X_noise, y, SVC(kernel='rbf', probability=True), svm_grid_rbf)
    nn_tuning_adam = tuning_grid_extra(X_noise, y, MLPClassifier(activation='relu', solver='adam', random_state=1), nn_grid_adam)
    nn_tuning_sgd = tuning_grid_extra(X_noise, y, MLPClassifier(activation='relu', solver='sgd', random_state=1), nn_grid_sgd)
    if nn_tuning_adam[1][1] > nn_tuning_sgd[1][1]:
        nn_tuning_opt = nn_tuning_adam
    else:
        nn_tuning_opt = nn_tuning_sgd

    return rf_tuning, svm_tuning, nn_tuning_opt


# %% Functions to evaluate the XAI explanations in terms of AOL, AOH, Fidelity and Stability
def evaluate_extra(exp, xai_method, n_feat_unrel, n_feat_rel, b):
    """Evaluates a given explanation 'exp' of xai_method in terms of AOL and AOH
    """
    beta_most_important = np.where(abs(b)==np.max(abs(b)))[0]

    if xai_method == 'lime':
        # for lime; here exp refers to the lime_exp
        ft_order = np.array(exp.as_map()[0])[:,0][::-1]
        
    elif xai_method == 'shap':
        # make list of features from lowest to highest attrbitution score
        ft_order = np.argsort(abs(exp.values[:,0]))
        
    # Below line makes a list with for each irrelevant feature score 0 if its attribution is one of the lowest n_feat_unrel
    # and score rank-n_feat_unrel if has higher attribution score than the lowest n_feat_unrel
    AOL = n_feat_unrel
    for j in range(n_feat_rel, n_feat_rel+n_feat_unrel):
        if j not in ft_order[:n_feat_unrel]:
            AOL -= 1
    AOL = AOL / n_feat_unrel

    AOH = len(beta_most_important)
    for j in beta_most_important:
        if j not in ft_order[::-1][:len(beta_most_important)]:
            AOH -= 1
    AOH = AOH / len(beta_most_important)

    return  AOL, AOH 


def all_metrics(X_train, X_test, classifier_trained, xai_method, n_feat_rel, n_feat_unrel, n_feat_list, b, y_train):
    """Returns the Fidelity, AOL and AOH scores, averaged over the observations in the test set"""
  
    def Fidelity_AO_one(X_test, index, classifier_trained, xai_method, n_feat_list, b):
        """Computes the Fidelity, AOL and AOH of one explanation for one observations at index 'index'"""     
        old_pred_prob = classifier_trained.predict_proba(X_test[index].reshape(1,-1))
        old_pred_class = old_pred_prob.argmax()

        old_pred_prob_comp = rf_comparing.predict_proba(X_test[index].reshape(1,-1))[0][old_pred_class]

        # Now run explainer model
        if xai_method == "lime":
            lime_exp = lime_explainer_sim.explain_instance(X_test[index], classifier_trained.predict_proba, num_features=n_feat_rel+n_feat_unrel, labels=(0,1))
            # collect the x most influential features
            # Select the features that have postive attribution towards the predicted class
            idx_positive_contr = np.where(np.array(lime_exp.as_map()[old_pred_class])[:,1] > 0) # already in order from most to least influential (abs) so no need for reordering here
            lime_features_pred_class = np.array(lime_exp.as_map()[0])[idx_positive_contr][:,0]
            
            # AOL and AOH
            AOL, AOH = evaluate_extra(lime_exp, xai_method, n_feat_unrel, n_feat_rel, b)


        elif xai_method == "shap":
            shap_exp = shap_explainer_sim(X_test[index].reshape(1,-1))[0]
            # collect the x most influential features
            # Select the features that have positive attribution towards the predicted class
            idx_positive_contr = np.where(np.array(shap_exp.values[:,old_pred_class]) > 0)[0]
            ft_order = np.argsort(abs(shap_exp.values)[:, old_pred_class])[::-1]
            ft_in_order_pred_class = np.array([f for f in ft_order if f in idx_positive_contr])
            # top_features = ft_in_order_pred_class[:n_feat].astype(int)
            
            # AOL and AOH
            AOL, AOH = evaluate_extra(shap_exp, xai_method, n_feat_unrel, n_feat_rel, b)

        def calc_mape(n_feat, X_test):
            """"Calculates the percentage change in prediction probability when the top n_feat are masked in an observation""""  
            X_test_copy = X_test.copy()

            if xai_method == 'lime':
                top_features = lime_features_pred_class[:n_feat].astype(int)
            elif xai_method == 'shap':
                top_features = ft_in_order_pred_class[:n_feat].astype(int)

            X_test_copy[:, top_features] = 0

            new_pred_prob_comp = rf_comparing.predict_proba(X_test_copy[index].reshape(1,-1))[0][old_pred_class]
            
            mape = abs(new_pred_prob_comp - old_pred_prob_comp)/old_pred_prob_comp
                
            return mape
        
        mape_list = []
        for n_feat in n_feat_list:
            mape_list.append(calc_mape(n_feat, X_test))
        
        return mape_list, AOL, AOH
    
    # the untuned Random Forest classifier is constructed to calculate changes in prediction probabilities
    rf_comparing = RandomForestClassifier()
    rf_comparing.fit(X_train, y_train)

    fidelity_list = []
    AOL = 0
    AOH = 0
    if xai_method == 'lime':
        lime_explainer_sim = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification')
    elif xai_method == 'shap':
        shap_explainer_sim = shap.PartitionExplainer(classifier_trained.predict_proba, shap.sample(X_train, 5))

    for ind in range(len(X_test)):
        fidelity, AOL_exp, AOH_exp = Fidelity_AO_one(X_test, ind, classifier_trained, xai_method, n_feat_list, b)
        fidelity_list.append(fidelity)
        AOL += AOL_exp
        AOH += AOH_exp

    return np.mean(fidelity_list, axis=0), score_2 / len(X_test), score_3 / len(X_test)


def metric_pipeline(X_noise, y, b, n_feat_list, clf_rf, clf_svm, clf_nn):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_noise, y, test_size = 0.33, random_state=0)
    clf_rf.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    clf_nn.fit(X_train, y_train)

    # RF
    rf_lime_all_metrics = all_metrics(X_train, X_test, clf_rf, 'lime', n_feat_rel, n_feat_unrel, n_feat_list, b, y_train)
    rf_shap_all_metrics = all_metrics(X_train, X_test, clf_rf, 'shap', n_feat_rel, n_feat_unrel, n_feat_list, b, y_train)
    # SVM
    svm_lime_all_metrics = all_metrics(X_train, X_test, clf_svm, 'lime', n_feat_rel, n_feat_unrel, n_feat_list, b, y_train)
    svm_shap_all_metrics = all_metrics(X_train, X_test, clf_svm, 'shap', n_feat_rel, n_feat_unrel, n_feat_list, b, y_train)
    # NN
    nn_lime_all_metrics = all_metrics(X_train, X_test, clf_nn, 'lime', n_feat_rel, n_feat_unrel, n_feat_list, b, y_train)
    nn_shap_all_metrics = all_metrics(X_train, X_test, clf_nn, 'shap', n_feat_rel, n_feat_unrel, n_feat_list, b, y_train)

    results = {'lime_rf': rf_lime_all_metrics, 'shap_rf': rf_shap_all_metrics,
               'lime_svm': svm_lime_all_metrics, 'shap_svm': svm_shap_all_metrics, 
               'lime_nn': nn_lime_all_metrics, 'shap_nn': nn_shap_all_metrics}

    return results



# %% Function to calculate the Stability of the explanations
def Stability_simulation(X_train, X_test, classifier, xai, n_top = 5, m=20):
    """"Calculates the Stability of the explanations from 'xai' on the test set over m iterations, considering the n_top most relevant features""""
    def explain_one_instance(X_test, index, classifier):
      """Returns the explanation of LIME for one observation at index 'index' in the test set, using trained classifier 'classifier'"""
        # Now run explainer model
        if xai == "lime":
            # lime_explainer_sim = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification')
            lime_exp = lime_explainer_sim.explain_instance(X_test[index], classifier.predict_proba, num_features=n_feat_rel+n_feat_unrel, labels=(0,1))
            # collect the x most influential features
            top_features = np.array(lime_exp.as_map()[0])[:n_top,0]
            feature_attributions = np.array(lime_exp.as_map()[0])[:n_top,1]

        return list(top_features.astype(int)), list(feature_attributions)
    

    def VSI(ft0, ft1):
        """Computes the VSI of two explanations"""
        return len(set(ft0).intersection(set(ft1)))/len(set(ft0).union(set(ft1)))
        # return len([word for word in ft0 if word in ft1])/len(ft0) # check if use other len maybe?
    
    def CSI(attr0, attr1, ft0, ft1):
        """Computes the CSI of two explanations"""
        ft_set = set(np.concatenate((ft0, ft1)))
        attr_diff_list = []
        for word in ft_set:
            if (word in ft0) and (word in ft1):
                attr_diff = 1 - abs(attr0[ft0.index(word)] - attr1[ft1.index(word)])/abs(attr0[ft0.index(word)])
            else: 
                attr_diff = 0 

            attr_diff_list.append(attr_diff)
        
        return np.array(attr_diff_list).mean()
    
    lime_explainer_sim = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification')

    VSI_ult_list = []
    CSI_ult_list = []

    for ind in range(len(X_test)):
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


def Stability_sim_par(X_noise, y, rf_clf, svm_clf, nn_clf, n_top_list = [1,3,5,10]):
  """"Uses parallel computing to compute the Stability of LIME's explanations in the test set using different classifiers and considering the top 1,3,5 and 10 most relevant features"""" 
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_noise, y, test_size = 0.33, random_state=0)
    rf_clf.fit(X_train, y_train)
    svm_clf.fit(X_train, y_train)
    nn_clf.fit(X_train, y_train)

    def compute_stability(X_train, X_test, rf_clf, svm_clf, nn_clf, n_top):
        stability_rf = Stability_simulation(X_train, X_test, rf_clf, 'lime', n_top, 5)
        stability_svm = Stability_simulation(X_train, X_test, svm_clf, 'lime', n_top, 5)
        stability_nn = Stability_simulation(X_train, X_test, nn_clf, 'lime', n_top, 5)

        return {'RF': stability_rf, 'SVM': stability_svm, 'NN': stability_nn, 'n_top': n_top}
    
    results = Parallel(n_jobs=-1)(delayed(compute_stability)(X_train, X_test, rf_clf, svm_clf, nn_clf, n_top) for n_top in n_top_list)

    return results


# %% Simulation
n_obs = 5000
n_feat_rel = 30
n_feat_unrel = 20
g_noise = 0.05

n_feat_list = [1,2,5,10]

# ----------------------------------------------------------------------------------------------------------

## Scenario 1: benchmark 
r = 0
b_factor = 20
intercept = 0

seed = 15
X_1, y_1, X_noise_1, cov_sim_1, b_1 = DGP(n_obs, n_feat_rel, n_feat_unrel, '303', r, g_noise, b_factor, intercept, seed)

rf_scn1 = RandomForestClassifier(n_estimators=1000, max_features=0.1, min_samples_leaf=1)
svm_scn1 = SVC(kernel='rbf', probability=True, C=100, gamma=0.01)
nn_scn1 = MLPClassifier(activation='relu', solver='adam', alpha=0.01, batch_size=128, hidden_layer_sizes=(50,), learning_rate_init=0.001, random_state=1)

all_metrics_new_scn1 = metric_pipeline(X_noise_1, y_1, b_1, n_feat_list, rf_scn1, svm_scn1, nn_scn1)
stability_scn1 = Stability_sim_par(X_noise_1, y_1, rf_scn1, svm_scn1, nn_scn1)


# ----------------------------------------------------------------------------------------------------------

## Scenario 2: correlatie 
r = 0.9
b_factor = 20
intercept = 0

seed = 34
X_2, y_2, X_noise_2, cov_sim_2, b_2 = DGP(n_obs, n_feat_rel, n_feat_unrel, '303', r, g_noise, b_factor, intercept, seed)

rf_scn2 = RandomForestClassifier(n_estimators=1000, max_features=0.1, min_samples_leaf=1)
svm_scn2 = SVC(kernel='rbf', probability=True, C=15, gamma=0.1)
nn_scn2 = MLPClassifier(activation='relu', solver='adam', alpha=0.01, batch_size=32, hidden_layer_sizes=(50,), learning_rate_init=0.001, random_state=1)

all_metrics_new_scn2 = metric_pipeline(X_noise_2, y_2, b_2, n_feat_list, rf_scn2, svm_scn2, nn_scn2)
stability_scn2 = Stability_sim_par(X_noise_2, y_2, rf_scn2, svm_scn2, nn_scn2)

# ----------------------------------------------------------------------------------------------------------

## Scenario 3: imbalance 
r = 0
b_factor = 20
intercept = -10

seed = 26
X_3, y_3, X_noise_3, cov_sim_3, b_3 = DGP(n_obs, n_feat_rel, n_feat_unrel, '303', r, g_noise, b_factor, intercept, seed)

rf_scn3 = RandomForestClassifier(n_estimators=400, max_features=0.35, min_samples_leaf=1)
svm_scn3 = SVC(kernel='rbf', probability=True, C=100, gamma=0.1)
nn_scn3 = MLPClassifier(activation='relu', solver='adam', alpha=0.01, batch_size=32, hidden_layer_sizes=(150,), learning_rate_init=0.001, random_state=1)

all_metrics_new_scn3 = metric_pipeline(X_noise_3, y_3, b_3, n_feat_list, rf_scn3, svm_scn3, nn_scn3)
stability_scn3 = Stability_sim_par(X_noise_3, y_3, rf_scn3, svm_scn3, nn_scn3)


# ----------------------------------------------------------------------------------------------------------

## Scenario 4: overlap (overlap 50%)
r = 0
b_factor = 0.5
intercept = 0

seed = 31
X_4, y_4, X_noise_4, cov_sim_4, b_4 = DGP(n_obs, n_feat_rel, n_feat_unrel, '303', r, g_noise, b_factor, intercept, seed)

rf_scn4 = RandomForestClassifier(n_estimators=800, max_features=0.1, min_samples_leaf=15)
svm_scn4 = SVC(kernel='rbf', probability=True, C=1, gamma=0.01)
nn_scn4 = MLPClassifier(activation='relu', solver='sgd', alpha=0.1, batch_size=128, hidden_layer_sizes=(150,), learning_rate_init=0.001, learning_rate='adaptive', random_state=1)

all_metrics_new_scn4 = metric_pipeline(X_noise_4, y_4, b_4, n_feat_list, rf_scn4, svm_scn4, nn_scn4)
stability_scn4 = Stability_sim_par(X_noise_4, y_4, rf_scn4, svm_scn4, nn_scn4)


# ----------------------------------------------------------------------------------------------------------

## Scenario 5: email data
r = 0.15
b_factor = 10
intercept = -2.1

seed = 4
X_5, y_5, X_noise_5, cov_sim_5, b_5 = DGP(n_obs, n_feat_rel, n_feat_unrel, '303', r, g_noise, b_factor, intercept, seed)

rf_scn5 = RandomForestClassifier(n_estimators=200, max_features=0.35, min_samples_leaf=1)
svm_scn5 = SVC(kernel='rbf', probability=True, C=70, gamma=0.1)
nn_scn5 = MLPClassifier(activation='relu', solver='sgd', alpha=0.01, batch_size=64, hidden_layer_sizes=(150,), learning_rate_init=0.01, learning_rate='adaptive', random_state=1)

all_metrics_new_scn5 = metric_pipeline(X_noise_5, y_5, b_5, n_feat_list, rf_scn5, svm_scn5, nn_scn5)
stability_scn5 = Stability_sim_par(X_noise_5, y_5, rf_scn5, svm_scn5, nn_scn5)

