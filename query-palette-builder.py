#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argcomplete, argparse
import json

parser = argparse.ArgumentParser(description='Iteractive prompt for palette querying/usage.')
parser.add_argument('rulesfile', type=argparse.FileType('r'),
                    help='A .rules file to be queried with questions about the palette')

argcomplete.autocomplete(parser)
args = parser.parse_args()


import os
import sys

import numpy as np
import pandas as pd


from prompt_toolkit import PromptSession

from palette_rule_analysis import yield_palettes
from graph_generator import rgb2hex
from graph_generator import rgb_to_hsv
from graph_generator import ciede2000_from_rgb


print("Loading rule database...")
db = pd.read_json(args.rulesfile)
filename=args.rulesfile.name.split('/')[-1].replace('.rules', '.gpl')
palette_data = []

print("Loading palette...")
for palette in yield_palettes(args.rulesfile.name.replace('.rules', '.gpl')):
    palette_data = [tuple(i) for i in palette['data']]



for label in ['antecedents', 'consequents']:
    db[label] = db[label].apply(lambda x: tuple(x))



def color_truename(index, full=False):
    if index < len(palette_data):
        color = palette_data[index]
        i = index
        rgb_list = palette_data
        if not full:
            return ("{: >3}"
                    "|" "{:.2f}|{:.2f}|{:.2f}"
                    #" " "{}"
                    .format(i,
                            *rgb_to_hsv(*rgb_list[i]),
                            #rgb2hex(*rgb_list[i]).upper()
                    ))
        else:
                    return ("{: >3}"
                            "|" "{:.2f}|{:.2f}|{:.2f}"
                            " " "{}"
                    .format(i,
                            *rgb_to_hsv(*rgb_list[i]),
                            rgb2hex(*rgb_list[i]).upper()
                    ))
    else:
        return "{:02}".format(index)

def parse_numbers(text):
    numbers = [''.join([i for i in x.strip() if i in '0123456789.'])
               for x in text.split(' ')]
    numbers = [float(i) if '.' in i else int(i) for i in numbers if len(i) > 0]
    return numbers


def rule_query(text):
    numbers = parse_numbers(text)
    index = tuple([int(i) for i in numbers])

    rules = db.copy()
    rules = rules[ rules['antecedents'].map(lambda x: all([i in x for i in index])
                                            and all([i in index for i in x]))
                   & rules['consequents'].map(lambda x: not any([i in x for i in index]))]
    if rules.empty:
        rules = db.copy()
        rules = rules[ rules['antecedents'].map(lambda x: all([i in x for i in index])
                                                and all([i in index for i in x]))
                       & rules['consequents'].map(lambda x: not any([i in x for i in index]))]

    if rules.empty:
        rules = db.copy()
        rules = rules[ rules['antecedents'].map(lambda x: all([i in x for i in index]))
                       & rules['consequents'].map(lambda x: not any([i in x for i in index]))]

    if rules.empty:
        rules = db.copy()
        rules = rules[ rules['antecedents'].map(lambda x: all([i in index for i in x]))
                       & rules['consequents'].map(lambda x: not any([i in x for i in index]))]


    rules.sort_values(by=['confidence', 'support', 'antecedents', 'consequents'], axis=0, inplace=True, ascending=True)

    if rules.empty:
        print("\t<empty>\n")
    else:
        rules = rules.values.tolist()
        rules = [(sorted(tuple(ant)), sorted(tuple(cons)), conf, sup) for ant, conf, cons, sup in rules]


        rules.sort(key=lambda x: (len(x[0]), 1/len(x[1]), x[2], x[0], x[1], x[3]))
        for previous, result, confidence, support in rules:
            print('\t{}\t→\t{}\t(C = {:4.2f}%, S = {:4.2f}%)'
                  .format(", ".join(map(str, previous)),
                          ", ".join(map(color_truename, result)),
                          confidence*100, support*100))

        print("   [Colors were:  {}]".format(", ".join(map(color_truename, index))))
        print("   [{} rules found]".format(len(rules)))


def sort_palette_query(text):
    mode = 'V'

    for md in ['H', 'S', 'V']:
        if md in text:
            mode = md

    text = text.strip().strip('sort palette_HSV')
    index = parse_numbers(text)

    table = ['H', 'S', 'V']
    getter = lambda x: x[table.index(mode)]
    index.sort(key=lambda i: getter(rgb_to_hsv(*palette_data[i])))

    for i, j in enumerate(index):
        print("   {:02}. {}".format(i, color_truename(j, full=True)))

def ciede_query(text):
    text = text.strip().strip('ciede2000 CIEDE')
    index = int(parse_numbers(text)[0])

    pl = list(range(len(palette_data)))

    c = [ciede2000_from_rgb(palette_data[index], palette_data[x]) for x in pl]
    pl.sort(key=lambda x: c[x], reverse=True)

    for j, (index, ciede) in enumerate([(p, c[p]) for p in pl]):
        print("   {:02}. {}  (ΔCIEDE={})".format(j, color_truename(index), ciede))



session = PromptSession('?' + filename + '> ')
print("   [{} rules found]".format(db.shape[0]))
while True:
    text = session.prompt()
    print("")

    if len(text) < 1:
        exit()

    processed = False
    parsers = [(['r ', 'ru ', 'rul ', 'rule '], rule_query),
               (['s ', 'sp ', 'sort_palette ', 'sort palette'], sort_palette_query),
               (['c ', 'ciede ', 'C ', 'CIEDE ', 'ciede2000 ', 'CIEDE2000 '], ciede_query),
    ]

    for (matching, querier) in parsers:
        if any([text.startswith(match) for match in matching]):
            processed = True
            querier(text)
            break

    if processed is False:
        pass
