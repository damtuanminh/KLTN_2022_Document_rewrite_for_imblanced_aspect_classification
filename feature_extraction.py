import sys
import numpy as np
import pandas as pd
from modules.preprocess import load_aspect_data_tmdt, preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def Chi2(inputs, outputs, num_aspects):
    if num_aspects == 6:
        categories = ['Ship', 'Gia', 'Chinh hang', 'Chat luong', 'Dich vu', 'An toan']
    else:
        categories = ['Cau hinh', 'Mau ma', 'Hieu nang', 'Ship', 'Gia', 'Chinh hang', 'Dich vu', 'Phu kien']

    cv = CountVectorizer()
    x = cv.fit_transform(inputs)

    y = []
    skb = [SelectKBest(chi2, k='all') for _ in range(num_aspects)]
    print(x.shape[0])
    for i in range(num_aspects):
        y.append([op[i] for op in outputs])
        _chi2 = skb[i].fit_transform(x, y[i])

        feature_names = cv.get_feature_names()
        _chi2_scores = skb[i].scores_
        _chi2_pvalues = skb[i].pvalues_

        chi2_dict = {'word': feature_names, 'score': list(_chi2_scores), 'pvalue': list(_chi2_pvalues)}
        df = pd.DataFrame(chi2_dict, columns=['word', 'score', 'pvalue'])
        df = df.sort_values('score', ascending=False)

        # with open(r"/Users/minhdam/Desktop/tech_shopee 2/data/chi2/data_{}/{}_chi2_dict/{}_{}.txt".format(str(sys.argv[1])[0:4], str(sys.argv[1]), str(sys.argv[1]), categories[i]), 'w', encoding='utf8') as f:
        #     for w, s, p in zip(df['word'], df['score'], df['pvalue']):
        #         f.write('{} \t {} \t {}\n'.format(w, s, p))
        return df