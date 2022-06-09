import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append("'m")
stop_words.append("'s")
stop_words.append("'re")
stop_words.append("'ve")

def chg_filename(org):
    org = org[:-4]
    tagged = pos_tag(word_tokenize(org))
    proc_sen = []
    for t in tagged:
        if t[1].startswith('N') or t[1].startswith('V') or t[1].startswith('J'):
            lemma = lemmatizer.lemmatize(t[0])
            if not lemma in stop_words:
                proc_sen.append(lemma)
    return ' '.join(proc_sen) + '.png'

if __name__ == '__main__':
    folder_path = './text2image/model/org_img_data'
    target_path = './text2image/model/img_data'
    file_name_list = os.listdir(folder_path)
    for file_name in file_name_list:
        org = folder_path + '/' + file_name
        new = target_path + '/' + chg_filename(file_name)
        os.rename(org,new)