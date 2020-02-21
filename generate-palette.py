from PIL import Image
import os
import pandas as pd
from PIL import ImageColor
from PIL import Image
import json
from sklearn import cluster
from sklearn import preprocessing
import numpy as np
import math
import pprint as Pprint

from common import pprint
from common import find_images
from common import yield_image_data
from common import dataset_dirs
from common import cached_image_data
from common import images_cache_path

palette_objects = []
number_images = sum([len(list(find_images(dataset_path))) for dataset_path in dataset_dirs])

json_data_path = images_cache_path

palette_objects = cached_image_data(dataset_dirs, json_data_path)


# Now we can finally do the fun analysis with the data

color_usage = {}

for palette in palette_objects:
    filepath = palette['filepath']
    colors = palette['colors']
    #raw_data = palette['raw_data']

    for color, quantity in colors:
        color = tuple(color)
        color_usage[color] = color_usage.get(color, 0) + 1
    # print(filepath)

#pprint(sorted([(key, value) for key, value in color_usage.items()], key=lambda x: color_usage[x[0]]))

number_hist = {}
for key, value in color_usage.items():
    number_hist[value] = number_hist.get(value, 0) + 1

pprint(number_hist)

from graph_generator import rgb2hex
from graph_generator import ciede2000_matrix_from_rgb

color_list = [color for color in color_usage.keys()]
color_list.sort()


# The following lines just prints a histogram of unique colors
my_number_hist = dict([x for x in tuple(list(number_hist.items()))])
my_color_list = list(tuple(color_list))
while len(my_number_hist) > 0:
    cur_min = 1000000
    for qtd in my_number_hist.keys():
        cur_min = min([cur_min, qtd])

    del my_number_hist[cur_min]
    my_color_list = [color for color in my_color_list if color_usage[color] > cur_min]
    print("There are currently {: 9} unique colors that appears more than {: 6} times"
          .format(len(list(set(my_color_list))), cur_min))

# color_usage = dict([(key, value)
#                     for key, value in color_usage.items()
#                     if value > 24])


color_list = [color for color in color_usage.keys() if color_usage[color] > 25 and color_usage[color] < 99999]
color_list.sort(key=lambda x: color_usage[x], reverse=True)

color_list = [color for i, color in enumerate(color_list) if (i % 1) == 0]

print(len(color_list))


# print('Generating occurence matrix...')

# R = []
# G = []
# B = []

# for color in color_list:
#     qtd = color_usage[color]
#     r_qtd = max(qtd, 600)
#     r_qtd = math.ceil(qtd/700)
#     r, g, b = color
#     for i in range(r_qtd):
#         R.append(r)
#         G.append(g)
#         B.append(b)

# color_list = list(zip(R, G, B))
# print(len(color_list))
# occurence_matrix = np.array(list(zip(R, G, B)))
# print(len(occurence_matrix))

# exit()
# abort()

print("Generating CIEDE2000 matrix...")
ciede_matrix = ciede2000_matrix_from_rgb(color_list, cached=True)
#print(ciede_matrix)

selected_matrix = ciede_matrix
#selected_matrix = occurence_matrix


print("Clustering matrix...")
def clustering(ciede_matrix, clusters=False):
    if len(ciede_matrix) < len(color_list):
        print("ciede_matrix is too short. Maybe it's a corrupted cache?")
        exit()

    if clusters is True:
        print("Normalizing matrix for clustering...")
        normalized_matrix = preprocessing.normalize(selected_matrix, norm='l2')

        #algorithm = cluster.AffinityPropagation(damping=0.95, max_iter=900, convergence_iter=280)
        algorithm = cluster.AgglomerativeClustering(n_clusters=63, linkage="ward")
        #algorithm = cluster.SpectralClustering(n_clusters=48)


        algorithm.fit(normalized_matrix)
        result = list(algorithm.labels_)
        print(result)
        print(max(result))


        categories = {}
        for category, color in zip(result, color_list):
            categories[category] = categories.get(category, []) + [color]

        #representants = [color_list[index] for index in algorithm.cluster_centers_indices_]
        representants = [sorted(category, key=lambda x:color_usage[x])[-1]  for category in categories.values()]

        return representants

    # Uses CIEDE2000 differences as a guide to choosing colors
    else:
        threshold = 10
        possible_index = set(range(len(color_list)))
        covered_colors = set([])
        colors = []


        while len(covered_colors) < len(color_list):
            cur_index = -1
            cur_row = None
            cur_score = 0
            for index, row in enumerate(ciede_matrix):
                if index in covered_colors:
                    continue
                else:
                    score = sum([1 if (x < threshold) else 0 for x in row ])
                    if (cur_row is None) or (score > cur_score):
                        cur_score = score
                        cur_index = index
                        cur_row = row

            if cur_row is not None:
                colors.append(color_list[cur_index])

                for i, v in enumerate(cur_row):
                    if v < threshold and i not in covered_colors:
                        print("Color {}/{} is covered by {}/{}"
                              .format(color_list[i], i, color_list[cur_index], cur_index))
                        covered_colors = covered_colors.union(set([i]))

                possible_index = possible_index - covered_colors

            print("There are {:04} remaining colors, {:04} are already covered"
                  .format(len(possible_index),
                          len(covered_colors)))



        return colors




representants = clustering(ciede_matrix, clusters=False)
representants.sort(key=lambda x: color_usage[x], reverse=True)
print(representants)
print(len(representants))

def write_gimp_file(colors, palette_path):
    # colors is a color list (each list's item is a 3-tuple [R, G, B])
    with open(palette_path, 'w') as write_file:
        write_file.write('GIMP Palette\n#\n')

        for color in colors:
            write_file.write('{: 3} {: 3} {: 3}\t#{}\n'
                             .format(color[0],
                                     color[1],
                                     color[2],
                                     rgb2hex(*color)))

        print('Palette generated.')

write_gimp_file(representants, 'pokemon.gpl')
#write_gimp_file(list(color_usage.keys()), 'zpokemon-all.gpl')
