from PIL import Image
from SelectiveSearch.segmentation.graph import build_graph, segment_graph
from SelectiveSearch.segmentation.smooth_filter import gaussian_grid, filter_image
from random import random
from numpy import sqrt
import numpy

# 用来计算rgb图像梯度
# img[0][x,y],x,y表示第x行第y列
def diff_rgb(img, x1, y1, x2, y2):
    r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
    g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
    b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
    return sqrt(r + g + b)

# 用来计算灰度图像的梯度
def diff_grey(img, x1, y1, x2, y2):
    v = (img[x1, y1] - img[x2, y2]) ** 2
    return sqrt(v)

def threshold(size, const):
    return (const / size)

# 根据forest生成结果图
def generate_image(forest, width, height):
    # 随机生成颜色
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in range(width*height)]

    # img = Image.new('RGB', (width, height))
    im_mask = numpy.zeros((height,width))
    # im = img.load()
    for x in range(height):
        for y in range(width):
            # 对每个像素，找到它属于的树
            comp = forest.find(x * width + y)
            # print(x)
            # print(y)
            # im[x, y] = colors[comp]
            im_mask[x, y] = comp
    return im_mask
    # return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

def graphbased_segmentation(img_path, neighbor, sigma, K, min_size):
    # neighbor = int(8)
    image_file = Image.open(img_path)
    # sigma = float(0.5)
    # K = float(800)
    # min_size = int(20)

    size = image_file.size
    # print('Image info: ', image_file.format, size, image_file.mode)

    # 生成高斯滤波算子
    grid = gaussian_grid(sigma)

    if image_file.mode == 'RGB':
        image_file.load()
        r, g, b = image_file.split()
        # 对r,g,b三个通道分别进行滤波(height,width),x行y列
        r = filter_image(r, grid)
        g = filter_image(g, grid)
        b = filter_image(b, grid)
        # print(r.shape)

        smooth = (r, g, b)
        diff = diff_rgb
    else:
        smooth = filter_image(image_file, grid)
        diff = diff_grey
    # print(smooth[0].shape)
    # 对图像中每个像素作为顶点建立图。
    graph = build_graph(smooth, size[0], size[1], diff, neighbor == 8)
    # 图像分割
    forest = segment_graph(graph, size[0]*size[1], K, min_size, threshold)
    image = generate_image(forest, size[0], size[1])
    # image.save("output2.jpg")
    # print('Number of components: %d' % forest.num_sets)

    return image
