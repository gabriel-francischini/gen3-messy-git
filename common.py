import os
from PIL import Image
from PIL import ImageColor
import pandas as pd
import json
import pprint as Pprint


pp = Pprint.PrettyPrinter(indent=4)
images_cache_path = 'images.json'


def pprint(x):
    pp.pprint(x)


dataset_dirs = [
    './data/dbsarchive/',
    './data/generation-3/',
    './data/spriters-resource/',
]


def find_images(basepath):
    for (dirpath, dirnames, filenames) in os.walk(basepath):
        for filename in filenames:
            if any([extension in filename.lower()
                    and len(filename.lower().strip('./').strip()) > 0
                    and '.' in filename
                    for extension in ['.bmp', '.gif', '.png']]):
                yield os.path.join(dirpath, filename)

def yield_image_data(dataset_dirs, with_raw_data=True, with_individual_cache=True):
    # Do some conversion between types
    imagepaths = []

    # Enforce that dataset_dirs is a list of paths
    if type(dataset_dirs) != type([]) and type(dataset_dirs) == type(""):
        dataset_dirs = [dataset_dirs]

    # Enforce that imagepaths is a list of paths to images
    for dataset_path in dataset_dirs:
        # Directly add images passed as args
        if os.path.isfile(dataset_path):
            imagepaths.append(dataset_path)
        # Recursively search non-image paths for images that may lay inside them
        else:
            for imagepath in find_images(dataset_path):
                imagepaths.append(imagepath)

    imagepaths.sort()
    # For every image (in)directly passed as arg
    for imagepath in imagepaths:
        cache_file = imagepath.rsplit('.', 1)[0] + '.json'

        if os.path.isfile(cache_file):
            with open(cache_file, "r") as read_file:
                obj = json.load(read_file)
                obj['colors'] = [(tuple(k), v) for k, v in obj['colors']]
                only_colors = [k for k, v in obj['colors']]

                max_w, max_h = obj['shape']
                del obj['shape']

                raw_data = obj['raw_data']
                raw_data = [raw_data[i:i+2] for i in range(0, len(raw_data), 2)]
                raw_data = [int('0x' + i, 16) for i in raw_data]

                obj['raw_data'] = {}
                x = 0
                y = 0

                for i in raw_data:
                    obj['raw_data'][(x, y)] = only_colors[i]

                    x += 1

                    if x > max_w:
                        y += 1
                        x = 0

                del raw_data

                # print((x, y), (max_w, max_h))
                # print((len(raw_data), len(obj['raw_data'])))
                # print(obj['raw_data'])
                # pprint(obj.keys())
                # exit()

                # if with_raw_data is True:
                #     obj['raw_data'] = [(tuple(k), tuple(v)) for k, v in obj['raw_data']]
                if with_raw_data is False:
                    del obj['raw_data']

                yield obj
                continue

        palette = {'filepath': imagepath}
        colors = {}

        # Key is (x, y) position, value is RGB color
        raw_data = {}

        im = Image.open(imagepath)
        rgb_im = im.convert('RGB')

        width, height = rgb_im.size

        # see: https://stackoverflow.com/questions/39548328/faster-method-of-looping-through-pixels
        pixLocation = [(y, x) for x in range(height) for y in range(width)]
        pixRGB = list(rgb_im.getdata())

        df = pd.DataFrame({'pixLocation': pixLocation, 'pixRGB': pixRGB})

        # see: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        for x in df.itertuples(index=False):
            pixRGB = x[-1]
            pixLocation = x[0]
            color = pixRGB
            colors[color] = colors.get(color, 0) + 1

            coords = pixLocation

            if with_raw_data is True:
                # Key is (x, y) position, value is RGB color
                raw_data[coords] = color


        palette['colors'] = list(colors.items())
        palette['raw_data'] = list(raw_data.items())

        if len(palette) < 255:
            # Makes a byte-sequence copy of the raw_data, for minimizing storage space
            copy = dict(list(palette.items()))
            only_colors = [k for k, v in palette['colors']]

            max_w, max_h = sorted([k for k, v in copy['raw_data']])[-1]

            copy['shape'] = (max_w, max_h)
            xy_converter = lambda x, y: (max_w + 1) * y + x

            copy['raw_data'] = sorted(copy['raw_data'], key=lambda z: xy_converter(*z[0]))
            copy['raw_data'] = [only_colors.index(v) for k, v in copy['raw_data']]

            copy['raw_data'] = "".join(map(lambda x: '{:02X}'.format(x), copy['raw_data']))

            with open(cache_file, "w") as write_file:
                json.dump(copy, write_file)

        if with_raw_data is False:
            del palette['raw_data']

        yield palette


def cached_image_data(dataset_dirs, json_data_path):
    palette_objects = []
    number_images = sum([len(list(find_images(dataset_path))) for dataset_path in dataset_dirs])

    if os.path.isfile(json_data_path):
        with open(json_data_path, "r") as read_file:
            print("Loading json...")
            palette_objects = json.load(read_file)
    else:
        image_counter = 0
        for dataset_path in dataset_dirs:
            for imagepath in find_images(dataset_path):
                for palette in yield_image_data(imagepath, with_raw_data=False):
                    image_counter += 1

                    # if image_counter >= 1000:
                    #     break

                    palette_objects.append(palette)

                    print("{:04} / {:04}: {}".format(image_counter, number_images, imagepath), flush=True)

        with open(json_data_path, "w") as write_file:
            print("Saving data to json...")
            json.dump(palette_objects, write_file)

    return palette_objects
