class Node:
    def __init__(self, parent, rank=0, size=1):
        # 母顶点
        self.parent = parent
        # 优先级
        self.rank = rank
        # 该点为母顶点时，区域顶点数量
        self.size = size

    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s)' % (self.parent, self.rank, self.size)

# 一个有向图由若干棵有向树构成生成森林
class Forest:
    def __init__(self, num_nodes):
        # 顶点列表
        self.nodes = [Node(i) for i in range(num_nodes)]
        # 分割区域数量
        self.num_sets = num_nodes

    def size_of(self, i):
        return self.nodes[i].size

    # 找到该顶点的母顶点
    def find(self, n):
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent

        self.nodes[n].parent = temp
        return temp

    # 顶点a所在区域和顶点b所在区域合并
    def merge(self, a, b):
        if self.nodes[a].rank > self.nodes[b].rank:
            self.nodes[b].parent = a
            self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
        else:
            self.nodes[a].parent = b
            self.nodes[b].size = self.nodes[b].size + self.nodes[a].size

            if self.nodes[a].rank == self.nodes[b].rank:
                self.nodes[b].rank = self.nodes[b].rank + 1

        self.num_sets = self.num_sets - 1

    def print_nodes(self):
        for node in self.nodes:
            print(node)

# 创建边，方向由(x,y)指向(x1,y1)，大小为梯度值
def create_edge(img, width, x, y, x1, y1, diff):
    # lamda:函数输入是x和y，输出是x * width + y
    vertex_id = lambda x, y: x * width + y
    w = diff(img, x, y, x1, y1)
    return (vertex_id(x, y), vertex_id(x1, y1), w)

# 生成图，对每个顶点，←↑↖↗创建四条边，达到8-邻域的效果
def build_graph(img, width, height, diff, neighborhood_8=False):
    graph = []
    for x in range(height):
        for y in range(width):
            if x > 0:
                graph.append(create_edge(img, width, x, y, x-1, y, diff))

            if y > 0:
                graph.append(create_edge(img, width, x, y, x, y-1, diff))

            if neighborhood_8:
                if x > 0 and y > 0:
                    graph.append(create_edge(img, width, x, y, x-1, y-1, diff))

                if x > 0 and y < width-1:
                    graph.append(create_edge(img, width, x, y, x-1, y+1, diff))

    return graph

def remove_small_components(forest, graph, min_size):
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a != b and (forest.size_of(a) < min_size or forest.size_of(b) < min_size):
            forest.merge(a, b)

    return  forest

def segment_graph(graph, num_nodes, const, min_size, threshold_func):
    weight = lambda edge: edge[2]

    forest = Forest(num_nodes)
    # 对所有边，根据其权值从小到大排序
    sorted_graph = sorted(graph, key=weight)
    # 初始化区域内部差列表
    threshold = [threshold_func(1, const)] * num_nodes

    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]

        if parent_a != parent_b and a_condition and b_condition:
            # print(parent_a)
            # print(parent_b)
            # print(weight(edge))
            # print(threshold[parent_a])
            # print(threshold[parent_b])
            forest.merge(parent_a, parent_b)
            a = forest.find(parent_a)
            # print(a)
            # 因为遍历时是从小到大遍历，所以如果合并，这条边的权值一定是新区域所有边最大的权值，即为该新区域的内部差
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size, const)

    return remove_small_components(forest, sorted_graph, min_size)
