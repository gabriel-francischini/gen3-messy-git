import os
import sys
from PIL import ImageColor
from PIL import Image

from common import pprint
from common import yield_image_data

from palette_rule_analysis import yield_palettes
from palette_rule_analysis import palette_translator
from palette_rule_analysis import palette_dirs




def find_files_with_extension(basepath, extensions):
    for (dirpath, dirnames, filenames) in os.walk(basepath):
        for filename in filenames:
            if any([extension in filename.lower()
                    and len(filename.lower().strip('./').strip()) > 0
                    and '.' in filename
                    for extension in extensions]):
                yield os.path.join(dirpath, filename)


recolorize_dir = './data/recolorize/'
output_dir = './data/recolorized'


for image in yield_image_data(recolorize_dir, with_raw_data=True):
    image_path = image['filepath']
    image_palette = [tuple(v) for v, c in image['colors']]
    pixels = dict([(tuple(k), tuple(v)) for k, v in image['raw_data']])
    max_w, max_h = sorted(list(pixels.keys()))[-1]

    for palette in sorted(yield_palettes(palette_dirs), key=lambda x: x['name']):
        palette_name = palette['name']
        palette = [tuple(v) for v in palette['data']]

        print("Recolorizing image {} for palette {}..."
              .format(os.path.basename(image_path), palette_name))

        translator = palette_translator(palette, image_palette)

        new_filename = os.path.basename(image_path).rsplit('.', 1)[0] + ' - ' + palette_name + '.png'
        new_filepath = os.path.join(output_dir, new_filename)

        newImg = Image.new('RGB', (max_w + 1, max_h + 1), "white")
        new_pixels = newImg.load()

        print("Recolorizing...", end="", flush=True)
        for (x, y), rgb in sorted(pixels.items()):
            sys.stdout.write('\b'*25)
            print("[{:04}, {:04}]".format(x, y), end="", flush=True)
            new_pixels[x, y] = translator[rgb]
        sys.stdout.write('\b'*25)
        print(' '*80, end="", flush=True)
        sys.stdout.write('\b'*120)

        newImg.save(new_filepath)
