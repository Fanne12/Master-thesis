This repository contains the Python code that was written to produce the results for my Master's thesis titled "The use of Explainable AI for Email Classification by Machine Learning Algorithms in the field of eDiscovery". The code is organized as follows:
1. "1. Preprocessing.py":   
   a. loads the email data of the TREC 2010 Legal interactive task into Python  
   b. performs data cleaning  
   c. performs data pre-processing steps  
2. "2. Tuning.py":  
   a. vectorizes the data using TF-IDF, Word2Vec and BERT, resulting in three vectorized datasets for each of the email datasets (Topic 303 and Topic 304)  
   b. performs hyperparameter tuning of the Random Forest, Support Vector Machine and Neural Network classifiers on the vectorized email datasets  
3. "3. Classification":  
   a. trains the classification models using the hyperparameter settings found by running "2. Tuning.py"  
   b. classifies the vectorized email datasets using the trained classification models and return their accuracy, f1-score, recall and precision  
4. "4. Simulation":  
   a. investigates the principal components, distribution and overlap of the Word2Vec vectorized email datasets  
   b. constructs five synthetic datasets  
   c. performs hyperparameter tuning of the Random Forest, Support Vector Machine and Neural Network classifiers on the synthetic datasets  
   d. trains the classifiers and performs the classification on the synthetic datasets  
   e. produces LIME and SHAP explanations for the predictoins on the synthetic datasets  
   f. evaluate the XAI explanations that were produced in terms of AOL, AOH, Fidelity and Stability  
5. "5. email_xai":  
   a. produces LIME and SHAP explanations for the predictions of the Word2Vec vectorized email datasets  
   b. evaluates the XAI explanations that were produces in terms of Fidelity and Stability  
