import math
import sys
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 显示进度条
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    s = "\r{}:[{}{}]{}%\t{}/{}".format(message,">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    # r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(s)
    sys.stdout.flush()

# 显示回归框
def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()