import argparse
import json
FILE_CONFIG = open('config.json')
CONFIG_MODEL = json.load(FILE_CONFIG)

parser = argparse.ArgumentParser(description='define config model.')
parser.add_argument('--select-model', type=str, default='mebeshopee',
                    help='model selection')
args = parser.parse_args()

specifications = CONFIG_MODEL[args.select_model]

print(specifications)