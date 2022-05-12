from vncorenlp import VnCoreNLP
import random
import fasttext
import json
from scipy.spatial import distance

RATIO = 0.7


def random_choice_elements(replace_word_list, n):
    try:
        return random.sample(replace_word_list, n)
    except:
        return replace_word_list


def create_new_review(review, replace_word_list):
    if len(replace_word_list) >= 1:
        flag = {}
        for i in replace_word_list:
            flag[i[0].replace('_', ' ')] = i[1]
        for i in range(len(review)):
            if review[i] in flag:
                review[i] = flag[review[i]]
    return review


def create_list_token_of_review(token_list):
    res = []
    for i in token_list:
        res.append(i[0].replace('_', ' '))
    return res


def find_pos_tag(st):
    if st in ['N']:
        st = 'noun'
    elif st in ['Vb', 'V']:
        st = 'verb'
    elif st in ['A']:
        st = 'adj'
    return st


def check_exits_in_dict(word, json_dict, postag):
    if word in json_dict[postag]:
        return 1
    return 0


def cosine_similarity(v1, v2):
    return 1 - distance.cosine(v1, v2)


def check_pos_tag(x):
    pos_tag_list = ['V', 'Vb', 'A']
    if x in pos_tag_list:
        return 1
    return 0


def combine_list(list1, list2):
    res = []
    for i in range(len(list1)):
        res.append([list1[i], list2[i]])
    return res


def token_review(review, annotator):
    tokens = annotator.tokenize(review)
    res = str(' '.join(tokens[0]))
    return res


def most_similarity(word, vocab, ft):
    if len(vocab) > 0:
        new_word = word
        maximum = 0
        v1 = ft.get_word_vector(word.replace('_', ''))
        for item in vocab:
            v2 = ft.get_word_vector(item.replace(' ', ''))
            if maximum < cosine_similarity(v1, v2):
                maximum = cosine_similarity(v1, v2)
                new_word = item
        return new_word
    else:
        return word


class GenerateData:
    def __init__(self):
        file = open('data/fdictionary.json', 'r')
        self.dictionary = json.loads(file.read())

    def generate(self, inputs, outputs, z, annotator, random_seed):
        _input = inputs
        _output = [output[z] for output in outputs]
        if random_seed == 23:
            ft = fasttext.load_model('data/tech_sg_model.bin')
        else:
            ft = fasttext.load_model('data/mebe_sg_model.bin')
        pos = sum(_output)
        neg = len(_output) - pos
        new_input = []
        new_output = []
        flag = False
        if RATIO * neg > pos != 0:
            for i in range(len(_output)):
                if _output[i] == 1:
                    new_input.append(_input[i])
            flag = True
            size = int(RATIO * neg) - pos
            random.seed(random_seed)
            new_input = random.choices(new_input, k=size)
            new_output = [1 for _ in range(len(new_input))]
            for x in range(len(new_input)):
                new_input[x] = self.represent(new_input[x], annotator, ft)
        if flag:
            _input.extend(new_input)
            _output.extend(new_output)
        return _input, _output

    def represent(self, review, annotator, ft):
        """

        :param annotator:
        :param a review review:
        :return: a new review:
        """
        pos_tag = annotator.pos_tag(review.replace('_', ' '))
        _review = review.replace('_', ' ')
        new_review = pos_tag[0].copy()
        replace_word_list = []
        for i in pos_tag[0]:
            if check_pos_tag(i[1]) and (len(i[0].split('_')) == 2):
                pos = find_pos_tag(i[1])
                if check_exits_in_dict(i[0].replace('_', ' '), self.dictionary, pos):
                    tmp_list = self.dictionary[pos][i[0].replace('_', ' ')]
                    new_word = i[0].replace('_', ' ')
                    if len(tmp_list) > 0:
                        new_word = most_similarity(i[0], tmp_list, ft)
                    if i[0].replace('_', ' ') != new_word:
                        replace_word_list.append([i[0], new_word])
        if len(replace_word_list) > 0:
            _review = ' '.join(create_new_review(create_list_token_of_review(new_review), replace_word_list))

        return token_review(_review, annotator)
