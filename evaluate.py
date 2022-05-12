import faulthandler; faulthandler.enable()
from sklearn import model_selection
from modules.evaluate import cal_aspect_prf
from modules.aspect.gb_model import MebeAspectGBModel
from modules.preprocess import load_aspect_data_tmdt, preprocess
import json
import argparse

FILE_CONFIG = open('config.json')
CONFIG_MODEL = json.load(FILE_CONFIG)

parser = argparse.ArgumentParser(description='define config model.')
parser.add_argument('--select-model', type=str, default='mebeshopee',
                    help='model selection')
args = parser.parse_args()

if __name__ == '__main__':

    model_info = CONFIG_MODEL[args.select_model]
    inputs, outputs = load_aspect_data_tmdt(model_info['path_data'], model_info['num_of_aspect'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(inputs, outputs, test_size=0.2, random_state=14)
    model = MebeAspectGBModel(model_info['num_of_aspect'])
    model.train(X_train, y_train, random_seed=model_info['random_seed'])
    predicts = model.predict(X_test)
    cal_aspect_prf(y_test, predicts, num_of_aspect=model_info['num_of_aspect'], verbal=True)
