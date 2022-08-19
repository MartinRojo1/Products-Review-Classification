import re
import nltk
from nltk.stem import PorterStemmer
import spacy


from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer


tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

##############################################################################################

def remove_html_tags(text):
    
    clear = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    text = re.sub(clear,'',text)

    text = text.lower()
    
    return text

##############################################################################################

##############################################################################################

def stem_text(text):

    porter = PorterStemmer()

    stem_sentence=[]

    for i in text.split():
   
        stem_sentence.append(porter.stem(i))

        stem_sentence.append(" ")
        
    text = "".join(stem_sentence)

    return text

##############################################################################################

##############################################################################################

def lemmatize_text(text1):
    text1 = text1.lower()

    for i in text1.split():

        doc = nlp(i)
  
        for token in doc:

            text1 = text1.replace(token.text,token.lemma_)

            text1 = text1.lower()

    return text1

##############################################################################################

##############################################################################################

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    for i in contraction_mapping:

        text = text.replace(i,contraction_mapping[i])
    
    return text

##############################################################################################

##############################################################################################

def remove_accented_chars(text):

    textR = ''

    for i in text:

        i = i.replace('á','a')
        i = i.replace('é','e')
        i = i.replace('í','i')
        i = i.replace('ó','o')
        i = i.replace('ú','u')

        textR += i

    return textR
  
##############################################################################################
##############################################################################################
##############################################################################################

def remove_special_chars(text, remove_digits=False):
    
    text = text.lower()
       
    text = re.sub(r"[\W]+",' ', text)
        
    text = text.strip()

    if remove_digits == False:

        text = re.sub(r'[0-9]+', '', text)

    return text

##############################################################################################

##############################################################################################

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
     
     
    text = text.lower()

    text = text.split()

    without_stop_words = [word for word in text if not word in stopwords]
        
    text = without_stop_words

    text1 = ''

    for i in without_stop_words:

        text1 += i

        text1 += ' ' 

    return text1

##############################################################################################

##############################################################################################   

def remove_extra_new_lines(text):
    
    text = " ".join(text.split())

    return text

##############################################################################################

##############################################################################################

def remove_extra_whitespace(text):
    
    text  = " ".join(text.split())   

    return text

##############################################################################################

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
