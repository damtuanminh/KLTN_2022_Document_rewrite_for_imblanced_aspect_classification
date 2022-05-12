import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import string
from models import Input, AspectOutput
from sklearn.feature_extraction.text import CountVectorizer
from datagen import GenerateData
punctuations = list(string.punctuation)
useless_labels = ['295', '296', '314', '315', '329', '330', '348', '349']


def is_nan(s):
    return s != s


def contains_punctuation(s):
    for c in s:
        if c in punctuations:
            return True
    return False


def contains_digit(w):
    for i in w:
        if i.isdigit():
            return True
    return False


def typo_trash_labeled(lst):
    for i in lst:
        if i in useless_labels:
            return True
    return False

def Chi2(inputs, outputs):
    cv = CountVectorizer()
    x = cv.fit_transform(inputs)
    skb = SelectKBest(chi2, k='all')
    y = outputs
    _chi2 = skb.fit_transform(x, y)

    feature_names = cv.get_feature_names()
    _chi2_scores = skb.scores_
    _chi2_pvalues = skb.pvalues_

    chi2_dict = {'word': feature_names, 'score': list(_chi2_scores), 'pvalue': list(_chi2_pvalues)}
    df = pd.DataFrame(chi2_dict, columns=['word', 'score', 'pvalue'])
    df = df.sort_values('score', ascending=False)
    return df


def generate_inputs(inputs, outputs, text_len, i, annotator, random_seed):
    inp, outp = [], []
    for ip, op in zip(inputs, outputs):
        text = str(ip).strip().split(' ')
        if len(text) <= text_len:
            if random_seed != 23:
                for j in range(len(text)):
                    if contains_digit(text[j].strip()):
                        text[j] = '0'
                for token in text:
                    if len(token) <= 1 or token.strip() in punctuations:
                        text.remove(token)
            ip = ' '.join(text)
            inp.append(ip)
            outp.append(op.scores)
    ninp, noutp = GenerateData().generate(inp, outp, i, annotator, random_seed)
    df = Chi2(ninp, noutp)
    return ninp, noutp, df
