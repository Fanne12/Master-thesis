# %% Import packages
import pandas as pd
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import extract_msg
from nltk.corpus import names
import contractions
from nltk.corpus import words

# %%
# ######################################################################################
# --------------------- Loading the annotation data ---------------------
# ######################################################################################

def load_annotations():
    """Load the annotations of the TREC 2010 Legal interactive task into a dataframe
    and select only the rows corresponding to topic 303 and 304"""
    main_path = r"C:\Users\fconijn\OneDrive - KPMG\Documents\Thesis\Code"
    
    # Import 2010 annotations
    df_2010_interactive = pd.read_excel(os.path.join(main_path,"Annotations 2010 Interactive.xlsx"), header=None)
    df_2010_interactive.rename(columns={0: 'Topic', 2: 'Docid', 3: 'Label'}, inplace=True)
    
    # Select topics 303 and 304
    df_2010_interactive = df_2010_interactive[(df_2010_interactive["Topic"]==303) | (df_2010_interactive["Topic"]==304)]

    return df_2010_interactive

# ######################################################################################
# --------------------- Loading the email data ---------------------
# ######################################################################################

def load_emails(df_2010_interactive):
    """Load the email data, corresponding to the emails with docid in df_2010_interactive to only select emails that are used in the TREC 2010 Legal interactive task"""
    main_path = r"C:\Users\fconijn\OneDrive - KPMG\Documents\Thesis\Code\Enron folders"

    # Docids to be filtered
    docids_to_filter = df_2010_interactive["Docid"].tolist() 

    # Collect the folder names of the 
    persons_folder_names = os.listdir(main_path)

    # Iterate through the folders
    # Create an empty DataFrame from the emails list
    df_emails_2010_I = pd.DataFrame({
        'File Name': [],
        'File Content': []
    })

    for folder in persons_folder_names:
        print(folder) # to check during running at which folder the program is
        path_sub_folder = os.listdir(os.path.join(main_path, folder))
        for sub_folder in path_sub_folder:
            email_list = []
            docid_list = []
            for docid in os.listdir(os.path.join(main_path, folder, sub_folder)):
                if docid.endswith('.txt') and docid.replace(".txt", "") in docids_to_filter:
                    with open(os.path.join(main_path, folder, sub_folder, docid), 'r') as email:
                        try: # try-except statement to check if all characters in the textfile can be decoded or 'read'
                            content = email.read()
                            email_list.append(content)
                            docid_list.append(docid)
                        except Exception as e:
                            print(f"An error occured: {e}")
            df_emails_2010_I = pd.concat([df_emails_2010_I, pd.DataFrame({'File Name': docid_list, 'File Content': email_list})])

    return df_emails_2010_I

# ######################################################################################
# --------------------- Cleaning the email data ---------------------
# ######################################################################################

def clean_emails(df_emails_2010_I):
    """Cleaning the email data and extracting the body of the emails"""
    # Remove .txt from the file names
    df_emails_2010_I["File Name"] = df_emails_2010_I["File Name"].str.replace(".txt", "")

    # Change column names for easier reference
    df_emails_2010_I.rename(columns={"File Name": 'Docid', "File Content": 'Email'}, inplace=True)
    df_emails_2010_I.reset_index(inplace=True, drop=True)

    # Extract 'Date', 'From', 'To', 'Subject' and 'Body'
    df_emails_2010_I["Date"] = df_emails_2010_I["Email"].str.extract(r'(Date:.*\n)')
    df_emails_2010_I["From"] = df_emails_2010_I["Email"].str.extract(r'(From:.*\n)')
    df_emails_2010_I["To"] = df_emails_2010_I["Email"].str.extract(r'(To:.*\n)')
    df_emails_2010_I["Subject"] = df_emails_2010_I["Email"].str.extract(r'(Subject:.*\n)')

    # 'Body' is the text between 'X-ZLID: zl-edrm-enron-v2-benson-r-665.eml' ('benson' in this case, but depends on mailbox) and 
    # ***********
    # EDRM Enron Email Data Set has been produced in EML, PST and NSF format by ZL Technologies, Inc. This Data Set is licensed under a Creative Commons Attribution 3.0 United States License <http://creativecommons.org/licenses/by/3.0/us/> . To provide attribution, please cite to "ZL Technologies, Inc. (http://www.zlti.com)."
    # ***********
    end_string_body = '\n\n***********\nEDRM Enron Email Data Set has been produced in EML, PST and NSF format by ZL Technologies, Inc. This Data Set is licensed under a Creative Commons Attribution 3.0 United States License <http://creativecommons.org/licenses/by/3.0/us/> . To provide attribution, please cite to "ZL Technologies, Inc. (http://www.zlti.com)."\n***********\n'
    end_string_body_escaped = re.escape(end_string_body)

    # Extract the body part including the 'X-ZLID' en '****' parts
    df_emails_2010_I["Body"] = df_emails_2010_I["Email"].str.extract(fr'(X-ZLID:.*{end_string_body_escaped})', flags=re.DOTALL)
    # Remove the 'X-ZLID part'
    df_emails_2010_I["Body"] = df_emails_2010_I["Body"].str.replace(r'(X-ZLID:.*\n)', '', regex=True)
    # Remove the '****' part
    df_emails_2010_I["Body"] = df_emails_2010_I["Body"].str.replace(end_string_body, '', regex=False)

    # Check how many of the remaining email bodies are empty and remove these
    df_emails_2010_I = df_emails_2010_I[df_emails_2010_I["Body"].str.strip().astype(bool)]
    df_emails_2010_I = df_emails_2010_I[df_emails_2010_I["Body"].notna()]
    df_emails_2010_I["Body"] = df_emails_2010_I["Body"].str.strip()
    df_emails_2010_I = df_emails_2010_I[df_emails_2010_I["Body"]!='']

    return df_emails_2010_I

# ######################################################################################
# --------------------- Adding the annotations to the email data ---------------------
# ######################################################################################

def add_annotations(df_annotations, df_emails):
    """Merging the annotations and emails dataframes"""  
    # Add the annotations for topics 303 and 304
    topics = [303, 304]

    for top in topics:
        df_emails = df_emails.merge(df_annotations[df_annotations["Topic"]==top][["Docid", "Label"]], on="Docid", how="left")
        df_emails.rename(columns={'Label': f'Topic {top}'}, inplace=True)

    return df_emails

 
# ######################################################################################
# --------------------- Pre-processing the email data  --------------------- 
# ######################################################################################
def preprocess_new(df_emails_cleaned):
    """Performs the pre-processing steps to the cleaned annotated email dataset"""
    # Make copy of dataframe's relevant columns to new dataframe to perform pre-processing steps 
    df_emails_preprocessed = df_emails_cleaned[["Docid","Body", "Topic 303", "Topic 304"]].copy(deep=True)

    # Turn all characters to lower case
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.lower()

    # Remove all lines 'to', 'from', 'sent', 'cc', ('subject'), 'original message', 'forwarded by'
    # Remove every 'word' containing @ and in between <> and containing at least 4 - (this also covers 'original message' and 'forwarded by')
    pattern = r"^(to:|from:|sent:|cc:).*$"
    df_emails_preprocessed['Body'] = df_emails_preprocessed['Body'].apply(lambda x: re.sub(pattern, '', x, flags=re.MULTILINE))
    df_emails_preprocessed['Body'] = df_emails_preprocessed['Body'].str.replace(r'.*-{4,}.*\n?', '', regex=True)
    df_emails_preprocessed['Body'] = df_emails_preprocessed['Body'].str.replace(r'\S*@\S*', '', regex=True)
    df_emails_preprocessed['Body'] = df_emails_preprocessed['Body'].str.replace(r'<[^>]*>', '', regex=True)
    # Remove http links
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace(r'http\S+', '', regex=True)
    # Remove remaining non alphabet/special characters (incl. punctuation removal); NLTK
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace(r'[^a-zA-Z\s\']', ' ', regex=True)
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace(r"(?<=\s)'|'(?=\s)", ' ', regex=True) # only remove ' when preceded or followed by whitespace, so it won't be deleted in for example it's, but will be deleted for 'fjfj'
    # Remove 1-letter words
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace(r'\b\w\b', '', regex=True)
    # Remove words strings containing 'com', 'cc' 
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace(r'\b(?:com|cc)\b', '', regex=True)

   # Extend contractions
    def extend_contractions(text):
        """Extend contractions in text"""
        return contractions.fix(text)
    
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].apply(extend_contractions)

    # Remove 's
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace("'s", '') # remove 's  as this showed problems when making word2vec embeddings (market's was not in the corpora for example)

    # Remove any name from the text 
    nltk.download("names")
    names_m = [i.lower() for i in names.words('male.txt')]
    names_f = [i.lower() for i in names.words('female.txt')]
    names_total = names_m + names_f + ['enron', 'usa']

    pattern = r'\b(?:' + '|'.join(names_total) + r')\b'
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].str.replace(pattern, '', regex=True)

    # Remove words of less than 4 letters if they are not actually words (often abbreviations)
    nltk.download("words")
    words_corpus = set(word.lower() for word in words.words())
    def filter_short_words(text):
        """Remove words of less than 4 letters if they are not actually words (often abbreviations)"""
        filtered_text = ' '.join(word for word in text.split() if len(word)>=4 or word.lower() in words_corpus)
        return filtered_text
    
    df_emails_preprocessed["Body"] = df_emails_preprocessed["Body"].apply(filter_short_words)


    # Tokenization; NLTK
    df_emails_preprocessed["Tokens"] = df_emails_preprocessed["Body"].apply(nltk.tokenize.WhitespaceTokenizer().tokenize)

    # Spell-checker; SpellChecker
    spell = SpellChecker(distance=1)

    def Correct_Spelling(token_list):
        """Corrects the spelling of words"""
        corrected = []
        for word in token_list:
            corrected.append(spell.correction(word))
        return corrected

    df_emails_preprocessed["Tokens"] = df_emails_preprocessed["Tokens"].apply(Correct_Spelling) # takes +/- 1,5 minutes to run

    # Remove None words
    def remove_None(token_list):
        """Remove None words"""
        no_None_token_list = [token for token in token_list if not token is None]
        return no_None_token_list

    df_emails_preprocessed["Tokens"] = df_emails_preprocessed["Tokens"].apply(remove_None)

    # Stop-word removal; NLTK
    nltk.download('stopwords') # download nltk stopwords list
    nltk_stopwords_list = stopwords.words('english') # contains 179 stopwords
    own_stopwords_list = ['to', 'from', 'cc', 'subject', 'forwarded', 'by', 'on', 'am', 'pm', 'original', 'message', 'sent', 'thanks', 'please', 'would', 'know', 'get', 'say', 'need', 'may', 'might', 'should', 'will', 'shall', 'dear']
    all_stopwords_set = set(nltk_stopwords_list + own_stopwords_list)
    
    def stop_word_remove(token_list): 
        """Remove the stop words from a list of tokens"""
        cleaned_token_list = [token for token in token_list if not token in all_stopwords_set]
        return cleaned_token_list

    df_emails_preprocessed["Tokens"] = df_emails_preprocessed["Tokens"].apply(stop_word_remove)

    # POS tagging and Lemmatization; WordNet database, NLTK
    nltk.download('wordnet')

    # The pos_tag() from nltk was trained on the Treebank corpus
    nltk.download('averaged_perceptron_tagger')

    # Tags from treebank (hence nltk) are different than the ones used for lemmatization (wordnet), so need to map the treebank tags to wordnet tags
    # get_wordnet_pos function is from https://codefinity.com/courses/v2/c68c1f2e-2c90-4d5d-8db9-1e97ca89d15e/026d736a-1860-4e3e-a915-b926ca2d9ed8/b9469e7f-9eb7-40ce-8941-2fb946782907
    def get_wordnet_pos(treebank_tag):
        """'Translates' the treebank tags to wordnet tags"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def Lemmatization(token_list):
        """Function to POS tag and lemmatize words"""
        lemmatizer = WordNetLemmatizer()
        postagged_token_list = pos_tag(token_list)
        lemmatized_list = []
        for token, tag in postagged_token_list:
            tag_wordnet = get_wordnet_pos(tag)
            lemmatized_list.append(lemmatizer.lemmatize(token, tag_wordnet))
        return lemmatized_list

    df_emails_preprocessed["Tokens"] = df_emails_preprocessed["Tokens"].apply(Lemmatization) # takes 50s to run

    df_emails_preprocessed["Pre-processed body"] = df_emails_preprocessed["Tokens"].apply(lambda x: ' '.join(x))

    # Remove emails with empty pre-processed body
    df_emails_preprocessed = df_emails_preprocessed[df_emails_preprocessed["Pre-processed body"] != '']

    return df_emails_preprocessed

