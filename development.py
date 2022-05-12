import pandas as pd
import pickle

if __name__ == '__main__':
    for i in range(8):
        df = pd.read_csv('data/raw_data/label_{}_tech_shopee.csv'.format(i), encoding="utf-8")
        data = df[['text', '_score']]
        new_data = []
        for k in range(len(df['text'])):
            new_data.append([df['text'][k], df['_score'][k]])

        # with open('data/vocab/vocab_{}.txt'.format(i),"w") as output:
        #     for row in new_data:
        #         output.write(str(row) + '\n')

        with open('data/vocab/tech_shopee_vocab_{}.txt'.format(i), 'wb') as internal_filename:
            pickle.dump(new_data, internal_filename)
        # vocab = list(set(' '.join([str(t).strip() for t in df['text']]).split()))
        # vocab.sort()
        # with open('data/vocab/mebe_shopee_vocab_{}.txt'.format(i), 'w', encoding="utf-8") as f:
        #     for w in vocab:
        #         f.write('{}\n'.format(w))
        # print(vocab)
    # vocab = []
    # for t in df['text']:
    #     try:
    #         vocab = list(set(' '.join([t.strip()]).split()))
    #     except:
    #         print(t)
    # vocab.sort()