# Built in python libs
import ast
import os
from time import time

import pprint
from pynput.keyboard import Key, Listener
import requests

# image manipulation
import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
QUERY_URL = 'https://api.warframe.market/v1/items/{}/orders?include=item'
PYTESSERACT_CONFIG = '--psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "'
PRICE_CACHE_TIME_SECONDS = 259200  # 3 days in seconds
CACHED_DIR = "C:\\Users\\josep\\PycharmProjects\\WFRelicRewardPricer\\cached-prices\\"
FILTERED_ITEMS = [
    'forma blueprint'
]
"""
number of relics: [ leftmost relic -> (x left, y left, x right, y right),
               second relic   -> (x left, y left, x right, y right) ] 
"""
SCREENSHOT_COORDS = {
    2: [(725, 432, 950, 460),
        (967, 432, 1192, 460)],
    3: [(605, 432, 825, 460),
        (845, 432, 1070, 460),
        (1085, 432, 1310, 460)],
    4: [(485, 432, 705, 460),
        (725, 432, 945, 460),
        (970, 432, 1190, 460),
        (1210, 432, 1435, 460)]
}

CURRENT_UI_THEME = 'BARUUK'
FILTERED_HSV_RANGES = {
    'BARUUK': [(0, 75, 93), (32, 255, 255)]
}


def cache_file_expired(file_path, current_unix_time):
    # Trust that that file is there as it was found below in os.listdir()
    return os.path.getmtime(file_path) + PRICE_CACHE_TIME_SECONDS < current_unix_time


def price_items(num_relics):
    screenshots = take_screenshots(num_relics=num_relics)
    parsed_items = get_items_from_images(screenshots)
    print(parsed_items)
    unique_items = set(parsed_items)

    items = filter_items(unique_items)
    # check for cached item prices before hitting WFM
    items_needing_pricing = []
    already_priced_items = []

    cached_price_files = os.listdir(CACHED_DIR)
    for item in items:
        formatted_name = format_name_with_underscore(item)
        file_name = formatted_name + '.txt'
        full_file_path = CACHED_DIR + file_name
        # verify that cache isn't older than 3 days
        if file_name in cached_price_files:
            if not cache_file_expired(full_file_path, current_unix_time=int(time())):
                f = open(full_file_path, 'r')
                already_priced_items.append(ast.literal_eval(f.read()))
                print('Already had price for {}.'.format(item))
                continue

            print('Had a price for {}, but was expired.'.format(item))
            items_needing_pricing.append(item)
        else:
            print('Did not have a price for {} cached.'.format(item))
            items_needing_pricing.append(item)

    # wfm wants items as volt_prime_neuroptics not volt_prime_neuroptics_blueprint
    items_needing_pricing_formatted = \
        [format_name_with_underscore(format_name_as_wfm_name(item)) for item in items_needing_pricing]

    wfm_responses = get_json_from_wfm(items_needing_pricing_formatted)
    item_prices = calc_wfm_stats(items_needing_pricing, wfm_responses)
    # TODO: get rid of this next line :(
    prices_to_write = item_prices.copy()
    item_prices.extend(already_priced_items)

    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    # Order the print results by most expensive item first
    s_items = sorted(item_prices, key=lambda item: item['lowest <=10 prices'][0], reverse=True)
    pp.pprint([{
        item['item_name']: item['lowest <=10 prices'][0:5] if len(item['lowest <=10 prices']) >= 5
        else item['lowest <=10 prices']
    } for item in s_items])

    # write out the prices to files
    if prices_to_write:
        print('Writing new prices to cache: ' + str(prices_to_write))
        for item in prices_to_write:
            file_name = CACHED_DIR + '_'.join(item['item_name'].split(' ')) + '.txt'
            f = open(file_name, 'w')
            f.write(str(item))
            f.close()

    print('--------------------------------------')


def fix_typos(name):
    return name.replace('bartel', 'barrel') \
        .replace('bikeprint', 'blueprint') \
        .replace('blueprin', 'blueprint') \
        .replace('blueprintt', 'blueprint') \
        .replace('carries', 'carrier') \
        .replace('fragot', 'fragor') \
        .replace('fagor', 'fragor') \
        .replace('ninkendi', 'ninkondi') \
        .replace('inyx', 'nyx') \
        .replace('prine', 'prime') \
        .replace('pame', 'prime')


def format_name_with_underscore(name):
    name = '_'.join(name.split(' '))
    return name


def format_name_as_wfm_name(name):
    # WFM doesn't store blueprint on the name of anything except actual warframe blueprints
    if 'systems' in name or 'chassis' in name or 'neuroptics' in name or 'harness' in name or 'wings' in name:
        name = name.replace(' blueprint', '')

    return name


def main():
    with Listener(on_release=on_release) as listener:
        listener.join()


def on_release(key):
    if key == Key.f2:
        price_items(2)
    if key == Key.f3:
        price_items(3)
    if key == Key.f4:
        price_items(4)


def take_screenshots(num_relics):
    if not num_relics:
        raise ValueError('Number of relics needs to be at least one to take a screenshot.')

    relics = []
    for relic in range(num_relics):
        relics.append(
            ImageGrab.grab(
                bbox=SCREENSHOT_COORDS[num_relics][relic]))

    return relics


def take_top_row_screenshot(num_relics, relic):
    coords = SCREENSHOT_COORDS[num_relics][relic]
    top_row_coords = (coords[0], coords[1]-25, coords[2], coords[3]-22)
    return ImageGrab.grab(bbox=top_row_coords)


def show(title, img, color=True):
    if color:
        plt.imshow(img[:,:,::-1]), plt.title(title), plt.show()
    else:
        plt.imshow(img, cmap='gray'), plt.title(title), plt.show()


def get_items_from_images(images):
    items = []

    for image in images:
        # Make the image readable by cv2 by converting to an np array
        np_image = np.array(image)
        # convert from rgb -> HSV in order to use the cv2.inRange() function to create a mask
        opencv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           FILTERED_HSV_RANGES[CURRENT_UI_THEME][0],
                           FILTERED_HSV_RANGES[CURRENT_UI_THEME][1])
        show('mask', mask, False)

        res = 255 - mask
        show('result', res, False)

        # grayscale_image = cv2.cvtColor(nm.array(image), cv2.COLOR_BGR2GRAY)
        # Use pytesseract OCR
        text = pytesseract.image_to_string(res, lang='eng', config=PYTESSERACT_CONFIG).lower()

        # TODO - With this filter is it still possible to get \n in results?
        text = text.replace('\n', ' ')

        # Fix any weird typos that I've seen while testing and can't improve mask on
        text = fix_typos(text)

        # If there was wordwrap on the item detect it and recursively add the top row
        # Specific case for 'neuroptics blueprint' because of 'wukong \n neuroptics blueprint
        if text.startswith('systems') or text.startswith('chassis') or text.startswith('neuroptics') \
                or text == 'blueprint' or text == 'limb' or text == 'string' or text == 'carapace':
            screenshot = take_top_row_screenshot(len(images), images.index(image))
            text = get_items_from_images([screenshot])[0] + ' ' + text

        items.append(text)
    return items


def filter_items(items):
    return [item for item in items if item not in FILTERED_ITEMS]


def get_json_from_wfm(item_names):
    # This function expects the name to already come in formatted
    json_results = []
    for item_name in item_names:
        query_url = QUERY_URL.format(item_name)
        print('Fetching Item prices from {}...'.format(query_url))
        r = requests.get(query_url)
        json_results.append(r.json())

    return json_results


def calc_wfm_stats(items, wfm_responses):
    results = []

    for item, wfm_response in zip(items, wfm_responses):
        # calculate stats for each item.
        online_orders = []
        if 'payload' not in wfm_response.keys():
            # no payload for item. was an error.
            continue

        for order in wfm_response['payload']['orders']:
            if order['user']['status'] == 'ingame' and order['order_type'] == 'sell':
                online_orders.append(order)

        # build a map of cost -> count for online orders
        online_price_map = {}
        for order in online_orders:
            cost = order['platinum']

            # keep track of how many at what prices
            if cost in online_price_map:
                online_price_map[cost] += 1
            else:
                online_price_map[cost] = 1

        # sort so the keys in the dictionary are in increasing order of price, then grab first x prices.
        sorted_price_map = sorted(online_price_map.items())
        num_prices_remaining = 10
        lowest_prices = []
        for (price, num_appearance) in sorted_price_map:
            for i in range(num_appearance):
                lowest_prices.append(price)

                num_prices_remaining -= 1
                if num_prices_remaining == 0:
                    break

            if num_prices_remaining == 0:
                break

        item_info = {
            'item_name': item,
            'lowest <=10 prices': lowest_prices,
            'online orders checked': len(online_orders)
        }

        results.append(item_info)

    return results


if __name__ == '__main__':
    main()
