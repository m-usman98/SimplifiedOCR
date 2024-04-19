from trdg.generators import (GeneratorFromStrings,)
from tqdm.auto import tqdm
import os
import random
from PIL.Image import Image
import csv

def random_string_data_generator(data_length, number, symbol, lang_char, min, max):
    random_data = []
    for i in range(data_length):
        X = ""
        fixed_random_length = random.randint(min, max)
        for i in range(fixed_random_length):
            choose_char = random.randint(0, 5)
            if choose_char == 0:
                len_number = random.randint(0, len(number)-1)
                X += number[len_number]

            elif choose_char == 1:
                len_lang_symbol = random.randint(0, len(symbol)-1)
                X += symbol[len_lang_symbol]

            else:
                len_lang_char = random.randint(0, len(lang_char)-1)
                X += lang_char[len_lang_char]
        random_data.append(X)
    return random_data


def get_fonts(font_pth):
    list_all_fonts = os.listdir(font_pth)
    get_font_full_pth = []
    for i in list_all_fonts:
        get_font_full_pth.append(os.path.join(font_pth, i))
    return get_font_full_pth


def write_csv(generated_data, dataset_name):
    header = ["filename", "words"]
    csv_dict = []
    for idx, i in enumerate(generated_data):
        csv_dict.append([f"{idx}.png", i])

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    with open(f"{dataset_name}/labels.csv", 'w', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\r")
        writer.writerow(header)
        writer.writerows(csv_dict)
    f.close()


if "__main__" == __name__:
    number = '0123456789'
    symbol = "!\"#$%&'()*+,-./€[]{}"
    lang_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    num_string_data = 40000
    min_length = 5
    max_length = 15
    num_images = 40000
    font_path = "./font/"
    dataset_name = "Train"
    get_generated_string_data = random_string_data_generator(num_string_data, number, symbol, lang_char,
                                                             min_length, max_length)
    get_all_fonts = get_fonts(font_path)

    generator = GeneratorFromStrings(
        strings=get_generated_string_data,
        random_blur=True,
        random_skew=True,
        image_mode="RGB",
        size=200,
        fonts=get_all_fonts,
        # width=800,
        background_type=random.randint(0, 4),
        text_color="red,black,yellow,blue,green,purple"
    )

    write_csv(get_generated_string_data, dataset_name)

    for counter, (img, lbl) in tqdm(enumerate(generator), total=num_images):
        if counter >= num_images:
            break
        Image.save(img, f'{dataset_name}/{counter}.png')
