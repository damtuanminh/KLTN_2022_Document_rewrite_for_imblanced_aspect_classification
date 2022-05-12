import pandas as pd
import fasttext
from calchi2 import generate_inputs
from models import AspectOutput, Input
from modules.models import Model
from sklearn.ensemble import GradientBoostingClassifier
from vncorenlp import VnCoreNLP


class MebeAspectGBModel(Model):
    def __init__(self, num_of_aspects):
        self.NUM_OF_ASPECTS = num_of_aspects
        self.vocabs = []
        self.annotator = VnCoreNLP("/Users/minhdam/Desktop/KLTN/code/VnCoreNLP-master/VnCoreNLP-1.1.1.jar",
                                   annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')
        self.models = [GradientBoostingClassifier(random_state=14) for _ in range(self.NUM_OF_ASPECTS - 1)]

    def _represent(self, inputs, i):
        """

        :param list of models.Input inputs:
        :return:
        """

        features = []
        for ip in inputs:
            _feature = [vocab[1] if str(vocab[0]) in str(ip).split(' ') else 0 for vocab in self.vocabs[i]]
            features.append(_feature)
        return features

    def chi2vocabs(self, ndf, i):
        vocab = []
        for k in range(ndf.shape[0]):
            if (ndf.iat[k, 1]) > 10:
                vocab.append([ndf.iat[k, 0], ndf.iat[k, 1]])
        if len(self.vocabs) == i:
            self.vocabs.append(vocab)

    def combine_data(self, input, output):
        _inputs, _outputs = [], []
        for k in range(self.NUM_OF_ASPECTS - 1):
            inputs, outputs = [], []
            df = pd.DataFrame({
                'text': input,
                'label': output
            })
            df = df.astype({'label': str})
            aspects = list(range(self.NUM_OF_ASPECTS - 1))
            for index, row in df.iterrows():
                text = row['text'].strip()
                inputs.append(Input(text))
                _scores = list(row['label'][1:-1].split(', '))
                scores = [int(i) for i in _scores[:self.NUM_OF_ASPECTS - 1]]
                outputs.append(AspectOutput(aspects, scores))
            _inputs.append(inputs)
            _outputs.append(outputs)
        return _inputs, _outputs

    def train(self, inputs, outputs, random_seed):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        _input, _output = self.combine_data(inputs.copy(), outputs.copy())
        finput = []
        foutput = []

        for k in range(self.NUM_OF_ASPECTS - 1):
            _inputs, _outputs, ndf = generate_inputs(_input[k], _output[k], 100, k, self.annotator, random_seed)
            self.chi2vocabs(ndf, k)
            finput.append(_inputs)
            foutput.append(_outputs)
            X = self._represent(finput[k], k)
            self.models[k].fit(X, foutput[k])

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = []
        for i in range(self.NUM_OF_ASPECTS - 1):
            Xn = self._represent(inputs.copy(), i)
            X.append(Xn)

        outputs = []
        predicts = [self.models[i].predict(X[i]) for i in range(self.NUM_OF_ASPECTS - 1)]
        for ps in zip(*predicts):
            labels = list(range(self.NUM_OF_ASPECTS))
            scores = list(ps)
            if 1 in scores:
                scores.append(0)
            else:
                scores.append(1)
            outputs.append(AspectOutput(labels, scores))

        return outputs
