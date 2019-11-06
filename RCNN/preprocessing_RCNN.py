from __future__ import division, print_function, absolute_import
import numpy as np
from SelectiveSearch.selectivesearch import selective_search
# import selectivesearch
import tools
import cv2
import config
import os
import math
import random


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


# IOU Part 1 判断两个区域是否有交集
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    # if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
    #     if_intersect = True
    # elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
    #     if_intersect = True
    # elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
    #     if_intersect = True
    # elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
    #     if_intersect = True
    if (xmin_a < xmax_b <= xmax_a or xmin_a < xmin_b <= xmax_a) and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif (xmin_a < xmax_b <= xmax_a or xmin_a < xmin_b <= xmax_a) and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif (xmin_b < xmax_a <= xmax_b or xmin_b < xmin_a <= xmax_b) and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif (xmin_b < xmax_a <= xmax_b or xmin_b < xmin_a <= xmax_b) and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    # if (xmin_b < xmax_a <= xmax_b and xmin_b < xmin_a <= xmax_b) and (ymin_b < ymax_a <= ymax_b and ymin_b <= ymin_a < ymax_b):
    #     if_intersect = False
    # if (xmin_a < xmax_b <= xmax_a and xmin_a < xmin_b <= xmax_a) and (ymin_a < ymax_b <= ymax_a and ymin_a <= ymin_b < ymax_a):
    #     if_intersect = False
    else:
        return if_intersect
    # 如果相交
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        # 求交集的宽度和高度
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        # 求交集的面积
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        # 交集/并集
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


# Clip Image
def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


# Read in data and save data for Alexnet
def load_train_proposals(datafile, num_clss, save_path, threshold=0.5, is_svm=False, save=False):
    fr = open(datafile, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    for num, line in enumerate(train_list):
        labels = []
        images = []
        rects = []
        tmp = line.strip().split(' ')
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img_path = tmp[0]
        img = cv2.imread(tmp[0])
        # 选择搜索得到候选框
        img_lbl, regions = selective_search(img_path, neighbor=8, scale=500, sigma=0.9, min_size=20)
        candidates = set()
        ref_rect = tmp[2].split(',')
        ref_rect_int = [int(i) for i in ref_rect]
        Gx = ref_rect_int[0]
        Gy = ref_rect_int[1]
        Gw = ref_rect_int[2]
        Gh = ref_rect_int[3]
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding small regions
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            # 截取目标区域
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            # Delete Empty array
            if len(proposal_img) == 0:
                continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
            # Check if any 0-dimension exist
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            # resize到227*227
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            images.append(img_float)
            # IOU
            iou_val = IOU(ref_rect_int, proposal_vertice)
            # x,y,w,h作差，用于boundingbox回归
            rects.append([(Gx-x)/w, (Gy-y)/h, math.log(Gw/w), math.log(Gh/h)])
            # propasal_rect = [proposal_vertice[0], proposal_vertice[1], proposal_vertice[4], proposal_vertice[5]]
            # print(iou_val)
            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if is_svm:
                # iou小于阈值，为背景，0
                if iou_val < threshold:
                    labels.append(0)
                elif  iou_val > 0.6: # 0.85
                    labels.append(index)
                else:
                    labels.append(-1)
            else:
                label = np.zeros(num_clss + 1)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
        if is_svm:
            ref_img, ref_vertice = clip_pic(img, ref_rect_int)
            resized_ref_img = resize_image(ref_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            img_float = np.asarray(resized_ref_img, dtype="float32")
            images.append(img_float)
            rects.append([0, 0, 0, 0])
            labels.append(index)
        tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))

        if save:
            if is_svm:
                # strip()去除首位空格
                np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'), [images, labels, rects])
            else:
                # strip()去除首位空格
                np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'),
                        [images, labels])
    print(' ')
    fr.close()


# load data
def load_from_npy(data_set, is_svm=False):
    if is_svm:
        images, labels, rects = [], [], []
        data_list = os.listdir(data_set)
        # random.shuffle(data_list)
        for ind, d in enumerate(data_list):
            i, l, r = np.load(os.path.join(data_set, d))
            images.extend(i)
            labels.extend(l)
            rects.extend(r)
            tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
        # print(rects[45])
        print(' ')
        return images, labels, rects
    else:
        images, labels = [], []
        data_list = os.listdir(data_set)
        # random.shuffle(data_list)
        for ind, d in enumerate(data_list):
            i, l = np.load(os.path.join(data_set, d))
            # print(i)
            # print(l)
            images.extend(i)
            labels.extend(l)
            tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
        print(' ')
        return images, labels
