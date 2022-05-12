import pandas as pd
import unicodedata
import regex as re
# from pyvi import ViTokenizer, ViPosTagger

from models import Input, AspectOutput


def load_aspect_data_tmdt(path, num_of_aspects):
    """

    :param path:
    :return:
    :rtype: (list of models.Input, list of models.AspectOutput)
    """
    inputs = []
    outputs = []
    df = pd.read_csv(path, encoding='utf-8')
    for _, r in df.iterrows():
        t = str(r['text']).strip()
        inputs.append(Input(t).__str__())

        labels = list(range(num_of_aspects))
        scores = [0 if r['aspect{}'.format(i)] == 0 else 1 for i in range(1, num_of_aspects + 1)]
        outputs.append(AspectOutput(labels, scores))

    return inputs, outputs


def preprocess(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    return inputs

