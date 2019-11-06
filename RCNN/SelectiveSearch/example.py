# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from SelectiveSearch.selectivesearch import selective_search


def main():

    img_path = "2.jpg"
    # loading astronaut image
    # img = skimage.io.imread(img_path)

    # perform selective search
    img_lbl, regions = selective_search(
        img_path, neighbor = 8 , sigma = 0.5, scale = 200, min_size = 20)

    # 创建候选框集合candidate
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()
