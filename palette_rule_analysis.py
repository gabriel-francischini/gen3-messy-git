import os
import sys
from graph_generator import read_gimp_palette
from graph_generator import ciede2000_from_rgb
from common import cached_image_data
from common import yield_image_data
from common import dataset_dirs
from common import pprint
from common import images_cache_path
from graph_generator import rgb_to_hsv


import numpy as np
import pandas as pd
import math
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from tqdm import tqdm


def find_palettes(basepath):
    for (dirpath, dirnames, filenames) in os.walk(basepath):
        for filename in filenames:
            if any([extension in filename.lower()
                    and len(filename.lower().strip('./').strip()) > 0
                    and '.' in filename
                    for extension in ['.gpl']]):
                yield os.path.join(dirpath, filename)


def yield_palettes(dataset_dirs):
    # Do some conversion between types
    palettepaths = []

    # Enforce that dataset_dirs is a list of paths
    if type(dataset_dirs) != type([]) and type(dataset_dirs) == type(""):
        dataset_dirs = [dataset_dirs]

    # Enforce that palettepaths is a list of paths to palettes
    for dataset_path in dataset_dirs:
        # Directly add palettes passed as args
        if os.path.isfile(dataset_path):
            palettepaths.append(dataset_path)
        # Recursively search non-palette paths for palettes that may lay inside them
        else:
            for palettepath in find_palettes(dataset_path):
                palettepaths.append(palettepath)

    # For every palette (in)directly passed as arg
    for palettepath in sorted(palettepaths, reverse=False):
        palette = read_gimp_palette(palettepath)
        yield {'name': os.path.basename(palettepath).split('.')[0], 'data': palette }



def palette_translator(palette, image_palette, accurate = True):
    copy_palette = list(tuple(palette))
    translator = {}

    for color in image_palette:
        if accurate is False:
            hcolor = rgb_to_hsv(*color)
            copy_palette.sort(key=lambda x: math.sqrt(abs(hcolor[0] - rgb_to_hsv(*x)[0]) ** 2 +
                                                            abs(hcolor[0] - rgb_to_hsv(*x)[0]) ** 2 +
                                                            abs(hcolor[0] - rgb_to_hsv(*x)[0]) ** 2),
                              reverse=True)
            #rgb = copy_palette[0]

            # Because CIEDE2000 is so expensive to compute, we are gonna to discard
            # the worst possible candidates. Sometimes (1/20? 1/40?) this gets the
            # wrong result, but I'm gonna choose a faster computation over a more
            # accurate one
            worst_n_candidates = max(int(len(copy_palette)/8), 10)
            remaining = copy_palette[worst_n_candidates:]

            remaining.sort(key=lambda x: ciede2000_from_rgb(color, x))
            ciede = remaining[0]

            # if ciede not in copy_palette[::-1][0:max(int(len(copy_palette)/8), 10)]:
            # #if ciede == rgb:
            #     print("Eureka! {} {}".format(rgb, ciede))
            # else:
            #     print("Bummer! {} {} {}".format(rgb, ciede, copy_palette[:4]))
            translator[color] = ciede
        else:
            copy_palette.sort(key=lambda x: ciede2000_from_rgb(color, x))
            translator[color] = copy_palette[0]


    return translator

def conform_to_palette(palette, raw_colors):
    image_palette = sorted(list(set(sorted([tuple(x) for x in raw_colors]))))
    translator = palette_translator(palette, image_palette)

    return [translator[tuple(x)] for x in raw_colors]


palette_dirs = './data/palettes/'

def main():
    images_palettes = []

    for image in cached_image_data(dataset_dirs, images_cache_path):
        image_palette = [tuple(x[0]) for x in image['colors']]
        images_palettes.append(image_palette)

    #images_palettes = images_palettes[::-1]

    print("Images loaded.")
    for palette in yield_palettes(palette_dirs):
        palette_name = palette['name']
        palette = palette['data']

        print("Calculating values for palette {}".format(palette_name))

        color_occurences = []
        limit = 10000
        # bar = tqdm(total=min(limit, len(images_palettes)),
        #            desc="PAL \"{}.gpl\"".format(palette_name))

        total = min(limit, len(images_palettes))
        print("\t\t0000/0000", end="", flush=True)
        #for i, image in enumerate(images_palettes)):
        for i, image in enumerate(yield_image_data(dataset_dirs, with_raw_data=True)):
            sys.stdout.write('\b'*40)
            print("\t\t{:04}/{:04}".format(i, total), end="", flush=True)
            #bar.update(1)
            if i >= limit:
                break

            image_palette = [tuple(x[0]) for x in image['colors']]
            translator = palette_translator(palette, image_palette, accurate=False)


            block_w = 31
            block_h = 31
            raw_data = dict(image['raw_data'])
            max_w, max_h =  sorted([k for k in raw_data.keys()])[-1]

            blocks = {}

            for k, v in raw_data.items():
                x, y = k
                color = translator[v]
                key = (int(max_w // block_w) * int(y // block_h)) + int(x // block_w)
                blocks[key] = blocks.get(key, []) + [color]


            for j, color_data in sorted(blocks.items()):
                sys.stdout.write('\b'*40)
                print("\t\t{:04}/{:04} [{:04}, {:04}]".format(i, total, j, (max_w // block_w) * ((max_h // block_h) + 1)),
                      end="", flush=True)

                conformed_palette = set(color_data)

                if len(conformed_palette) >= 4:
                    row = [1 if color in conformed_palette else 0 for color in palette]
                    color_occurences.append(row)


            # for start_x in range(0, max_w, block_w):
            #     end_x = start_x + block_w
            #     for start_y in range(0, max_h, block_h):
            #         end_y = start_y + block_h
            #         sys.stdout.write('\b'*40)
            #         print("\t\t{:04}/{:04} [{:04}, {:04}]".format(i, total, start_x, start_y),
            #               end="", flush=True)
            #         color_data = [raw_data[(x, y)]
            #                       for x in range(start_x, end_x)
            #                       for y in range(start_y, end_y)
            #                       if (x, y) in raw_data]

            #         color_data = set([translator[x] for x in color_data])
            #         conformed_palette = color_data
            #         if len(conformed_palette) >= 4:
            #             row = [1 if color in conformed_palette else 0 for color in palette]
            #             color_occurences.append(row)


            # conformed_palette = conform_to_palette(palette, image_palette)
            # conformed_palette = set(conformed_palette)

            # row = [1 if color in conformed_palette else 0 for color in palette]
            # color_occurences.append(row)
        sys.stdout.write('\b'*40)
        print(' '*80, end="", flush=True)
        sys.stdout.write('\b'*120)


        # Some sprites are multiple sprites glued together, so they use the
        # entire palette. For association rules mining, we want to look on sprites
        # that only uses a small portion of the palette. GBA's palette
        # is limited to 15 usable colors (plus color 0 for transparency).
        color_occurences = [group for group in color_occurences if sum(group) <= min(math.ceil(len(palette)*0.78), 33)]
        remaining = len(color_occurences)


        color_occurences = np.array(color_occurences)
        color_occurences = np.reshape(color_occurences, (remaining, len(palette)))
        color_occurences = pd.DataFrame(color_occurences, dtype=np.bool_)
        color_occurences = color_occurences.rename(dict([(i, v) for i, v in enumerate(palette)]),
                                                   axis=1)

        min_support = 0.01
        if len(palette) < 87:
            min_support = 0.03
        elif len(palette) > 87:
            min_support = 0.01
        #print("Calculating apriori...")
        frequent_colorsets = apriori(color_occurences, min_support=min_support,
                                     #use_colnames=True
        )

        shape = frequent_colorsets.shape
        extra = ""
        colorsets_limit = 40000

        # Sometimes there are way more rows than my computer is able to handle
        if shape[0] > colorsets_limit:
            frequent_colorsets.sort_values('support', inplace=True, ascending=False)
            frequent_colorsets.drop(frequent_colorsets.tail(shape[0]-colorsets_limit).index,inplace=True)
            extra = " (clipped to {}, i.e. support={})".format(frequent_colorsets.shape,
                                                             frequent_colorsets.iloc[-1]['support'])

        print("\t\t→    Frequent colorsets totalize: {} {}".format(shape, extra))


        #pprint(frequent_colorsets)
        #exit()

        #print("Calculating association rules...")
        rules = association_rules(frequent_colorsets, metric="confidence", min_threshold=0.01)
        rules = rules[(rules['lift'] > 1)]

        for column in ['antecedent support', 'consequent support', 'leverage', 'conviction', 'lift']:
            rules.drop(column, inplace=True, axis=1)


        rules.sort_values(by=['confidence', 'support'], axis=0, inplace=True, ascending=False)

        #pprint(rules)
        #pprint(rules.head())
        print("\t\t→    Rules totalize: {}".format(rules.shape))

        with open('./data/palettes/' + palette_name + '.rules', "w") as write_file:
            write_file.write(rules.to_json())

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', 'fooprof')
    #main()
