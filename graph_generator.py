from ciede2000 import ciede2000
from ciede2000 import rgb2lab
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import pandas as pd
import functools
import itertools
from graphviz import Graph
import colorsys
import math
import os
import json


def rgb2hex(r, g, b):
    def clamp(x):
        return max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

def inverse_rgb(rgb):
    inverse = (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])
    return inverse

def average_rgb(rgb_A, rgb_B):
    rint = lambda x: int(round(x))
    diff = lambda x, y: rint(math.fabs(x-y)/2)
    diff_pos = lambda x: diff(rgb_A[x], rgb_B[x])
    return tuple([diff_pos(x) for x in [0, 1, 2]])


def round_by_step(value, step):
    return round(float(value) / step) * step

def rgb2gray_rgb(rgb):
    #l = 0.21 * rgb[0] + 0.72 * rgb[1] + 0.07 * rgb[2]
    l = sum(rgb)/3
    l = int(round_by_step(l, 250))
    if l < 0:
        l = 0
    if l > 255:
        l = 255
    return (l, l, l)

def rgb_to_hsv(r, g, b):
    R, G, B = colorsys.rgb_to_hsv(r, g, b)
    B = B/100
    return (R, G, B)

def read_gimp_palette(filename):
    data = []
    with open(filename, 'r') as datafile:
        firstline = datafile.readline()
        if firstline.strip() != "GIMP Palette":
            raise TypeError("It's not a GIMP Palette file")
        for line in datafile:
            if line.startswith('#'):
                continue
            if len(line.strip()) < 3:
                continue
            items = [x.strip() for x in line.split() if len(x.strip()) > 0]
            rgb = tuple([int(x) for x in items[:3] if x.isdigit()])
            if len(rgb) > 3:
                rgb = rgb[:3]
            if len(rgb) == 0:
                continue
            if len(rgb) < 3:
                # print(line)
                # print(items)
                # print(rgb)
                print(line)
                print(items)
                print(rgb)
                raise KeyError("Can't parse line")
            if rgb != (0, 0, 0): # Skip placeholder colors
                data.append(rgb)
    data_order = tuple(data)
    data = set(data)
    data = sorted(list(data), key=lambda x: data_order.index(x))
    return data

def ciede2000_from_rgb(rgb_A, rgb_B):
    return ciede2000(rgb2lab(rgb_A), rgb2lab(rgb_B))

def matrix_from_rgb_comparator(rgb_list, comparator=ciede2000_from_rgb):
    matrix = []
    for color in rgb_list:
        matrix.append(list(map(functools.partial(comparator, color),
                               rgb_list)))
    return matrix

def ciede2000_matrix_from_rgb(rgb_list, cached=False):
    # Creates a NxN matrix of CIEDE2000 differences
    #print("entry")
    #print(tuple(map(rgb2lab, rgb_list)))

    cache_file = 'ciede200_matrix.npdata'

    if cached is True and os.path.isfile(cache_file):
        with open(cache_file, "r") as read_file:
            cached_matrix = json.load(read_file)
            matrix = np.array(cached_matrix)
            del cached_matrix
            if matrix.shape == (len(rgb_list), len(rgb_list)):
                return matrix
            else:
                del matrix

    py_list = np.array(list(itertools.chain.from_iterable(list(map(rgb2lab, rgb_list)))), dtype=np.float32)
    py_list = np.reshape(py_list, (len(rgb_list), 3))
    #print(py_list.shape)
    #print("py_list")


    #py_list = np.transpose([np.tile(py_list, len(py_list)), np.repeat(py_list, len(py_list))])

    #print(py_list.shape)
    #print("cartesian")
    # a = pd.DataFrame(py_list, dtype=np.float64)
    # b = pd.DataFrame(py_list, dtype=np.float64)

    # a['key'] = 0
    # b['key'] = 0
    # print(b)

    # c = a.merge(b, how='outer', on='key')

    # print(c)

    # The following DOES work
    py_list = pd.DataFrame(py_list, dtype=np.float64)
    py_list['key'] = 0.0
    py_list = pd.merge(py_list, py_list, how='outer', on='key', sort=True)
    py_list.drop(axis=1, columns='key', inplace=True)


    # py_list['cide'] = np.vectorize(lambda a, b, c, d, e, f: a+b+c+d+e+f
    # )(py_list['0_x'],py_list['1_x'],py_list['2_x'],  py_list['0_y'],py_list['1_y'],py_list['2_y'])

    #py_list = py_list.apply(lambda x: 1, axis=1, result_type='reduce')
    #py_list = py_list.apply(lambda x: 1, axis=1)

    #new_list = []

    # for i, row in py_list.iterrows():
    #     print(row)

    #py_list = py_list.to_numpy(copy=True)



    #print(py_list)

    #cartesian = pd.merge(a, b, on='_tmpkey').drop('_tmpkey', axis=1)


    #cartesian = np.array(itertools.product(py_list, py_list))
    #print("cartesian")

    res = py_list.apply(lambda x:
                        ciede2000
                        ((x[0], x[1], x[2]), (x[3], x[4], x[5])), axis=1,
                        #broadcast=False,
                        raw=True,
                        #reduce=True,
                        result_type='reduce')
    #print(res)
    del py_list

    res = np.reshape(res.values, (len(rgb_list), len(rgb_list)))

    #print(res)
    #print(type(cartesian))
    #data_frame = pd.DataFrame(index=
    #exit()

    # Saves the cache file
    if cached:
        with open(cache_file, "w") as write_file:
            x = res.tolist()
            json.dump(x, write_file)
            del x

    return res
    return matrix_from_rgb_comparator(rgb_list, ciede2000_from_rgb)

def mst_matrix_from_matrix(matrix):
    # Minimum Spanning Tree
    matrix = csr_matrix(matrix)
    tcsr = minimum_spanning_tree(matrix)
    return [[y for y in x] for x in tcsr.toarray().astype(float)]

def calculate_threshold(rgb_list, matrix_function=ciede2000_matrix_from_rgb):
    tcsr = mst_matrix_from_matrix(matrix_function(rgb_list))
    return functools.reduce(max, [max(i) for i in tcsr])


def view_graph(rgb_list, matrix_function=ciede2000_matrix_from_rgb,
               filename='CIEDE2000', verbose=False, render=True,
               threshold_calculator=calculate_threshold, colorize=lambda x: x):
    matrix = matrix_function(rgb_list)
    threshold = threshold_calculator(rgb_list, matrix_function)
    graph = Graph('G', filename=filename+'.gv', format='png',
                  graph_attr={"fontname": "Ubuntu Mono"},
                  edge_attr={"fontname": "Ubuntu Mono"},
                  node_attr={"fontname": "Ubuntu Mono"})

    def replace_color(color):
        if color <= threshold:
            return color
        else:
            return 0
    filtered_matrix = [list(map(replace_color, x)) for x in matrix]

    edges = set()
    for i in range(len(rgb_list)):
        for j in range(len(rgb_list)):
            # Sanitize input
            if filtered_matrix[i][j] != filtered_matrix[j][i]:
                for k, l in [(i, j), (j, i)]:
                    filtered_matrix[k][l] = max(filtered_matrix[i][j],
                                                filtered_matrix[j][i])
            if filtered_matrix[i][j] > 0:
                edges.add(tuple(sorted((i, j))))

    nodes_list = set()
    nodes = {}
    for i, j in edges:
        for i in [i, j]:
            nodes_list.add(rgb_list[i])
    for i in range(len(rgb_list)):
        nodes[i] = ("Index {: >3}\nRGB {: >3} {: >3} {: >3}\nHSV {:.2f} {:.2f} {:.2f}\nHEX {}"
                    .format(i,
                            *rgb_list[i],
                            *rgb_to_hsv(*rgb_list[i]),
                            rgb2hex(*rgb_list[i]).upper()))

    for i in range(len(rgb_list)):
        if rgb_list[i] in nodes_list:
            graph.node(nodes[i],
                       shape='square',
                       style='filled',
                       color=rgb2hex(*colorize(rgb_list[i])),
                       fontcolor=rgb2hex(*rgb2gray_rgb(inverse_rgb(colorize(rgb_list[i])))))

    connections_histogram = {}
    for i, j in edges:
        graph.edge(nodes[i], nodes[j],
                   label='<<font color="#f44336"><b>{:.2f}</b></font>>'
                   .format(round_by_step(filtered_matrix[i][j], 0.01)),
                   _attributes={
                       "penwidth": "{}".format(
                           10/math.log(filtered_matrix[i][j], 2)),
                       "color": "{}:{}".format(rgb2hex(*colorize(rgb_list[i])),
                                               rgb2hex(*colorize(rgb_list[j]))),
                   })
        for i in [i, j]:
            connections_histogram[i] = connections_histogram.get(i, 0) + 1

    max_edges = max(list(connections_histogram.values()))
    max_digits = int(len(str(max_edges)))
    graph.attr(label=r'\n\n{}, threshold={:.2f}, nodes={}, edges={}'
               .format(filename, round_by_step(threshold, 0.01),
                       len(nodes_list), len(edges)))
    graph.attr(fontsize='44')

    if verbose:
        edges_copy = set(tuple(edges))
        for i, j in list(edges):
            edges_copy.add((j, i))
        for i, j in list(sorted(edges_copy,
                                key=lambda x: (connections_histogram[x[0]], x))):
            print(("{} ({: >" + str(max_digits) + "} con.)" " -- {: >5.2f} --"
                   "{} ({: >" + str(max_digits) + "} con.)")
                  .format(nodes[i].replace('\n', ' '), connections_histogram[i],
                          filtered_matrix[i][j],
                          nodes[j].replace('\n', ' '), connections_histogram[j]))
        print(' '*40)
        for k, v in list(sorted(connections_histogram.items(),
                                key=lambda x: x[1], reverse=True)):
            #print("{} <-> {}".format(nodes[k].replace('\n', ' '), v))
            if verbose:
                print("{: >3} {: >3} {: >3}\t{}"
                      .format(*rgb_list[k], rgb2hex(*rgb_list[k]).upper()))

    if render:
        graph.render()
    return filtered_matrix

if __name__ == '__main__':

    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir("./") if isfile(join("./", f))]
    for filepath in files:
        if not filepath.endswith('.gpl'):
            continue
        filename = filepath.replace('.gpl', '')
        print('Processing "{}"...'.format(filename))
        colors = tuple(read_gimp_palette(filepath))
        threshold = max(calculate_threshold(colors), 20)
        filtered_matrix = view_graph(colors, verbose=False, render=False)
        mst = mst_matrix_from_matrix(filtered_matrix)
        view_graph(colors, matrix_function=lambda x: mst, verbose=False,
                   filename='{}-ramp'.format(filename),
                   render=True, threshold_calculator=lambda x,y: threshold)

        # GBA can have up to 512 unique colors, we gonna limit it to 64-ish

        #if len(colors) < 64:
        if True:
            view_graph(colors, filename='{}-diagram'.format(filename),
                       verbose=False,
                       render=True, threshold_calculator=lambda x,y: threshold)

        # def only_luminance(rgb):
        #     h, s, v = colorsys.rgb_to_hsv(*rgb)
        #     return colorsys.hsv_to_rgb(h, 0, v)

        # view_graph(colors, filename='{}-value-diagram'.format(filename),
        #            verbose=False,
        #            render=True, threshold_calculator=lambda x,y: threshold,
        #            colorize=only_luminance)

    # thresholds = set()
    # for i in filtered_matrix:
    #     list(map(thresholds.add, i))
    # thresholds = [i for i in thresholds if i > 0]
    # thresholds.sort()

    # for i in range(len(thresholds)):
    #     view_graph(colors,
    #                filename=('CIEDE2000-{:0>'
    #                          + str(len(str(len(thresholds))))
    #                          + '}').format(i),
    #                render=True, threshold_calculator=lambda x, y: thresholds[i])

    print("Done.")
