import h5py
from PIL import Image
import numpy as np
import cv2 as cv


def convert_weather_value_to_px(left_border, value):
    if isinstance(value, float):
        value *= 100
        left_border *= 100
        left_border, value = int(left_border), int(value)

    g = 0 if value >= 0 else 255
    shifted_value = abs(value) - left_border
    r = shifted_value // 256
    b = shifted_value % 256
    return [r, g, b]


def convert_snp_to_px(value):
    r, g, b = 0, 0, 0
    if value == -1:
        r = 255
    elif value == 0.5:
        g = 255
    elif value == 2:
        b = 255
    return [r, g, b]


def get_weather_information(filename):
    with h5py.File(filename, "r") as f:
        weather_info = []
        weather_groups = list(f.keys())
        weather_groups[0], weather_groups[3] = weather_groups[3], weather_groups[0]

        # ['month', 'doy', 'geo_id', 'dl', 'rain', 'srad', 'tmax', 'tmin', 'year']
        for group in weather_groups:
            tmp = []
            for dset in f[group]:
                tmp.append(dset[0])
            weather_info.append(np.array(tmp))
        weather_info = np.array(weather_info).transpose()
        return weather_info


def get_vigna_information(filename):
    with h5py.File(filename, "r") as f:
        keys = list(f.keys())
        plant_groups = keys[:7] + keys[11:]
        for key in ('response_EM', 'month', 'gr_names'):
            plant_groups.remove(key)

        vigna_info = []
        # ['doy', 'geo_id', 'gr_covar', 'response_R1', 'species', 'year']
        for group in plant_groups:
            tmp = []
            for dset in f[group]:
                if group == 'species':
                    tmp.append(dset.decode('UTF-8'))
                elif group == 'gr_covar':
                    tmp.append(dset)
                else:
                    tmp.append(dset[0])
            vigna_info.append(np.array(tmp))
        vigna_info = list(map(list, zip(*vigna_info)))
        summer_plants, winter_plants = [], []
        for vigna in vigna_info:
            summer_plants.append(vigna) if vigna[0] < 240 else winter_plants.append(vigna)
        return summer_plants, winter_plants
'''
функция считывания png через (numpy, opencv) того размера который есть
брать массив фалйы и заполнять квадрат теми данные которыми она ссчитала ищ предыдщуего изображения и записывает в новое изображени
создать новую сетку
'''

def aio_transform(filename_vigna, filename_weather):
    weather = get_weather_information(filename_weather)
    summer_planted_vigna, winter_planted_vigna = get_vigna_information(filename_vigna)

    all_vigna = summer_planted_vigna + winter_planted_vigna
    for plant in summer_planted_vigna:
    #for plant in all_vigna:
        pixels = []
        snp_pixels = []
        for snp in plant[2]:
            snp_pixels.append(convert_snp_to_px(snp))
        pixels.append(snp_pixels[:5])
        pixels.append(snp_pixels[5:10:])
        pixels.append(snp_pixels[10:])

        for day in weather:
            if (day[-1] == plant[-1]) and (day[2] == plant[1]) and (plant[0] - 5 <= day[1] < plant[0] + 20):
                day_pixels = []
                for i in range(3, 8):
                    day_pixels.append(convert_weather_value_to_px(0, day[i]))
                pixels.append(day_pixels)
        pixels = np.array(pixels, dtype=np.uint8)
        new_image = Image.fromarray(pixels, mode="RGB").rotate(90, expand=True)
        new_image.save('AIO_summer/' + str(plant[4]) + '_' + str(plant[1]) + '_' + str(plant[3]) + '.png')


def cv_saliency_map():
    image = cv.imread("/Users/mariia/Desktop/snap.png")
    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv.imshow("Image", image)
    cv.imshow("Output", saliencyMap)
    cv.waitKey(0)
