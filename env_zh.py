"""
Useful function for interaction with the environment rrt*
@author: Yulin Li
@email: yline@connect.ust.hk
"""
# env_zh.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

class Env:
    """
    定义环境类，包括障碍物、边界、起点、终点和碰撞检测等。
    """
    def __init__(self, xI, xG, name):
        """
        初始化环境。
        :param xI: 起点坐标 (x, y)
        :param xG: 终点坐标 (x, y)
        :param name: 环境名称（用于绘图标题）
        """
        self.delta = 0.3  # 碰撞检测的安全边界扩展距离
        self.x_range = (0, 50)  # x轴范围
        self.y_range = (0, 30)  # y轴范围
        self.obs_boundary = self.obs_boundary()  # 获取环境边界障碍物
        self.obs_circle = self.obs_circle()      # 获取圆形障碍物
        self.obs_rectangle = self.obs_rectangle() # 获取矩形障碍物
        self.xI = xI  # 起点
        self.xG = xG  # 终点
        self.obs_vertex_list = []  # 存储矩形障碍物（含扩展边界）的顶点列表
        self.get_obs_vertex()      # 计算并存储矩形障碍物的顶点

    # obstacle pos and numbers could be adjusted as you want
    # 障碍物的位置和数量可以根据需要调整
    @staticmethod
    def obs_boundary():
        """
        定义环境的边界障碍物。
        格式: [x, y, width, height]
        """
        obs_boundary = [
            [0, 0, 1, 30],    # 左边界
            [0, 30, 50, 1],   # 上边界
            [1, 0, 50, 1],    # 下边界 (注意 x 坐标从 1 开始，宽度 50 到达 x=51，与右边界对齐)
            [50, 1, 1, 30]    # 右边界 (注意 y 坐标从 1 开始，高度 30 到达 y=31，与上边界对齐)
        ]
        # 注意：这里的边界定义方式可能导致角点处有微小空隙或重叠，取决于具体算法实现细节
        # 更常见的做法是定义四条线段或一个大的外框然后内部挖空。
        # 但基于原代码，我们保留这种矩形定义。
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        """
        定义矩形障碍物。
        格式: [左下角x, 左下角y, width, height]
        """
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        """
        定义圆形障碍物。
        格式: [圆心x, 圆心y, 半径r]
        """
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]
        return obs_cir

    @staticmethod
    def get_dist(start, end):
        """
        计算两个节点之间的欧氏距离。
        :param start: 起始节点 (Node对象)
        :param end: 结束节点 (Node对象)
        :return: 距离
        """
        return math.hypot(end.x - start.x, end.y - start.y)

    @staticmethod
    def cross(p1, p2, p3):
        """
        计算向量 (p1->p2) 和 (p1->p3) 的二维叉乘。
        用于判断点 p3 相对于线段 p1p2 的方向。
        :param p1, p2, p3: 点坐标 [x, y]
        :return: 叉乘结果 (大于0在左侧, 小于0在右侧, 等于0在线上)
        """
        # 向量 v1 = p2 - p1
        l1 = [p2[0] - p1[0], p2[1] - p1[1]]
        # 向量 v2 = p3 - p1
        l2 = [p3[0] - p1[0], p3[1] - p1[1]]
        # 二维叉乘: l1.x * l2.y - l1.y * l2.x
        return np.cross(l1, l2)

    def plot_grid(self, name):
        """
        绘制环境地图，包括边界、障碍物、起点和终点。
        :param name: 图表标题
        """
        fig, ax = plt.subplots() # 创建图表和坐标轴

        # 绘制边界 (黑色)
        for (ox, oy, w, h) in self.obs_boundary:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        # 绘制矩形障碍物 (灰色)
        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        # 绘制圆形障碍物 (灰色)
        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        # 绘制起点 (蓝色方块)
        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        # 绘制终点 (绿色方块)
        plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        plt.title(name) # 设置标题
        plt.axis("equal") # 保持x,y轴比例一致

    @staticmethod
    def plot_visited(nodelist, animation):
        """
        绘制 RRT* 算法访问过的节点（树的边）。
        :param nodelist: 包含所有树节点的列表
        :param animation: 是否启用动画效果 (缓慢绘制)
        """
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent: # 如果节点有父节点，则绘制连接线
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g") # 绿色实线
                    # 允许在动画过程中按 'escape' 键退出
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                lambda event:
                                                [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0: # 每绘制10条边暂停一下，产生动画效果
                        plt.pause(0.001)
        else: # 不启用动画，直接绘制所有边
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    @staticmethod
    def plot_path(path):
        """
        绘制最终找到的路径。
        :param path: 包含路径点坐标 [(x, y), ...] 的列表
        """
        if len(path) != 0:
            # 绘制路径 (红色粗实线)
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(0.01) # 短暂停顿以确保路径显示
        plt.show() # 显示最终绘制结果

    def get_obs_vertex(self):
        """
        计算并存储所有矩形障碍物的顶点（考虑了安全边界 delta）。
        每个障碍物存储为 [左下, 右下, 右上, 左上] 四个顶点坐标。
        """
        delta = self.delta # 安全边界
        for (ox, oy, w, h) in self.obs_rectangle:
            # 计算扩展后的矩形顶点
            vertex_list = [
                [ox - delta, oy - delta],           # 左下角 (v1)
                [ox + w + delta, oy - delta],       # 右下角 (v2)
                [ox + w + delta, oy + h + delta],   # 右上角 (v3)
                [ox - delta, oy + h + delta]        # 左上角 (v4)
            ]
            self.obs_vertex_list.append(vertex_list)

    def is_inside_obs(self, node):
        """
        检查一个节点是否位于任何障碍物内部（包括安全边界）。
        :param node: 要检查的节点 (Node 对象)
        :return: True 如果在障碍物内, False 否则
        """
        delta = self.delta

        # 检查是否在圆形障碍物内
        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        # 检查是否在矩形障碍物内 (使用预先计算的扩展顶点)
        for (v1, v2, v3, v4) in self.obs_vertex_list:
            # v1 左下, v2 右下, v3 右上, v4 左上
            if v1[0] <= node.x <= v2[0] and v1[1] <= node.y <= v3[1]: # AABB 碰撞检测
                return True

        # 检查是否在环境边界障碍物内 (也考虑 delta)
        for (x, y, w, h) in self.obs_boundary:
            # 需要检查节点是否在扩展后的边界矩形内
            # (注意边界的定义方式，这里假设边界本身就是障碍区)
            if (x - delta <= node.x <= x + w + delta and
                y - delta <= node.y <= y + h + delta):
                 # 更精确的边界检查可能需要区分是哪个边界条
                 # 但对于简单的“是否在边界障碍区内”判断，这种扩展矩形检查是可行的
                 # 检查点是否在边界矩形内（考虑delta）
                 # 修正边界检查逻辑：
                 is_in_horizontal = (x <= node.x <= x + w) and (y - delta <= node.y <= y + h + delta)
                 is_in_vertical = (x - delta <= node.x <= x + w + delta) and (y <= node.y <= y + h)
                 
                 # 如果点在原始矩形区域内，或者在因为delta扩展出的区域内
                 # 一个简化的检查方式：点是否在扩展后的大矩形内
                 # 这里使用更简单的逻辑：如果点非常靠近边界定义的矩形（包括delta），则认为是内部
                 if x <= node.x <= x + w and y <= node.y <= y + h: # 在原始边界矩形内
                     return True
                 # 检查是否在扩展区域内，这比较复杂，取决于边界的精确定义
                 # 原代码的逻辑似乎是检查点是否在一个比原始边界矩形稍大的区域内
                 # 我们保留原意，但需注意这可能不够精确
                 if (x - delta <= node.x <= x + w + delta and
                     y - delta <= node.y <= y + h + delta):
                     # 这个检查有点宽泛，可能会将边界外的点误判
                     # 考虑一个点 (0.5, 15)。它在左边界 [0,0,1,30] 的扩展区域内
                     # (0-0.3 <= 0.5 <= 1+0.3) and (0-0.3 <= 15 <= 30+0.3) -> True
                     # 这里的目的是防止节点太靠近“墙壁”
                     # 重新审视原代码逻辑:
                     # 0 <= node.x - (x - delta) <= w + 2 * delta and 0 <= node.y - (y - delta) <= h + 2 * delta
                     # 这等价于：
                     # x - delta <= node.x <= x + w + delta  AND  y - delta <= node.y <= y + h + delta
                     # 这就是检查点是否在扩展后的矩形区域内。
                     if (x - delta <= node.x <= x + w + delta and
                         y - delta <= node.y <= y + h + delta):
                          # 但是，这样检查对于细长的边界条来说太宽泛了
                          # 例如，对于左边界 [0,0,1,30]，它会覆盖 x 在 [-0.3, 1.3] 的区域
                          # 也许原意只是确保节点不“穿透”边界定义的线？
                          # 暂时保留这个逻辑，因为它在原代码中存在
                          # 但在实际应用中可能需要更精细的边界处理
                          
                          # 修正：直接使用原代码逻辑，它检查点是否在扩展后的AABB内
                          if 0 <= node.x - (x - delta) <= w + 2 * delta \
                             and 0 <= node.y - (y - delta) <= h + 2 * delta:
                              return True


        return False # 不在任何障碍物内

    def is_intersect_circle(self, origin, dire, center, radius):
        """
        检查从 origin 出发、方向为 dire 的线段（长度为 ||dire||）是否与圆形障碍物相交。
        使用投影法。
        :param origin: 线段起点 [x, y]
        :param dire: 线段方向向量 [dx, dy] (终点 = origin + dire)
        :param center: 圆心 [cx, cy]
        :param radius: 半径 r
        :return: True 如果相交, False 否则
        """
        # 线段函数: P(t) = origin + t * dire, 其中 t 在 [0, 1] 之间

        # 计算方向向量的模长平方
        d2 = np.dot(dire, dire)
        if d2 == 0: # 如果起点和终点相同，线段长度为0，不相交
            return False

        # 计算圆心到起点的向量 center - origin
        origin_to_center = [center[0] - origin[0], center[1] - origin[1]]

        # 计算 origin_to_center 在 dire 上的投影长度 t
        # t = dot(origin_to_center, dire) / dot(dire, dire)
        t = np.dot(origin_to_center, dire) / d2

        # 检查投影点是否在线段内部 (0 <= t <= 1)
        if 0 <= t <= 1:
            # 计算投影点坐标
            intersect = [origin[0] + t * dire[0], origin[1] + t * dire[1]]
            # 计算投影点到圆心的距离
            dist_to_center = math.hypot(intersect[0] - center[0], intersect[1] - center[1])
            # 如果距离小于等于半径 + 安全距离，则认为相交
            if dist_to_center <= radius + self.delta:
                return True
        
        # 即使投影点不在线段内 (t < 0 或 t > 1)，线段的端点也可能在圆内或非常接近圆
        # 这个检查在 is_collision 函数中通过检查端点 is_inside_obs 来处理
        # 因此这里只处理线段“划过”圆的情况

        return False

    def is_intersect_line(self, start, end, v1, v2):
        """
        检查线段 (start, end) 是否与线段 (v1, v2) 相交。
        使用叉乘法判断。
        :param start, end: 线段1 的端点 [x, y]
        :param v1, v2: 线段2 的端点 [x, y]
        :return: True 如果相交, False 否则
        """
        # 快速排斥实验 (可选，原代码似乎在 is_collision 中做了)
        # ...

        # 跨立实验 (使用叉乘)
        # cross(start, end, v1) 和 cross(start, end, v2) 异号
        # cross(v1, v2, start) 和 cross(v1, v2, end) 异号
        # 等于0表示点在线段上，也算相交
        if self.cross(start, end, v1) * self.cross(start, end, v2) <= 0 \
           and self.cross(v1, v2, start) * self.cross(v1, v2, end) <= 0:
            return True
        return False

    def is_collision(self, s, e):
        """
        检查从节点 s 到节点 e 的路径是否会与任何障碍物发生碰撞。
        :param s: 起始节点 (Node 对象)
        :param e: 结束节点 (Node 对象)
        :return: True 如果发生碰撞, False 否则
        """
        # 检查起点或终点本身是否在障碍物内部
        if self.is_inside_obs(s) or self.is_inside_obs(e):
            return True

        start = [s.x, s.y] # 起点坐标
        end = [e.x, e.y]   # 终点坐标
        direction = [end[0] - start[0], end[1] - start[1]] # 方向向量

        # 检查是否与矩形障碍物相交
        for (v1, v2, v3, v4) in self.obs_vertex_list:
            # 快速排斥实验 (AABB包围盒检查)
            # 检查线段 (start, end) 的包围盒是否与障碍物 (v1,v2,v3,v4) 的包围盒重叠
            if max(start[0], end[0]) >= v1[0] \
               and v2[0] >= min(start[0], end[0]) \
               and max(start[1], end[1]) >= v1[1] \
               and v3[1] >= min(start[1], end[1]):
                # 如果包围盒重叠，再进行精确的线段相交检查
                # 检查 (start, end) 是否与矩形的对角线相交
                # 注意：矩形障碍物是实心的，应该检查是否与四条边相交
                # 但检查两条对角线相交是一种简化，可以检测出大部分穿越情况
                # 更严谨的方法是检查与四条边的相交：(v1,v2), (v2,v3), (v3,v4), (v4,v1)
                # 原代码检查对角线 (v1,v3) 和 (v2,v4)
                if self.is_intersect_line(start, end, v1, v2) or \
                   self.is_intersect_line(start, end, v2, v3) or \
                   self.is_intersect_line(start, end, v3, v4) or \
                   self.is_intersect_line(start, end, v4, v1):
                    return True
                # 原代码的对角线检查逻辑：
                # if self.is_intersect_line(start, end, v1, v3) or self.is_intersect_line(start, end, v2, v4):
                #     return True
                # 改为检查四条边更准确


        # 检查是否与圆形障碍物相交
        for (x, y, r) in self.obs_circle:
            # 注意：这里检查的是线段与圆的相交，不是点是否在圆内（起点终点已检查过）
            if self.is_intersect_circle(start, direction, [x, y], r):
                return True

        # 检查是否与环境边界障碍物相交 (作为矩形处理)
        # 注意：边界的处理方式与普通矩形障碍物类似，但边界是“墙”
        # 这里的实现假设边界也是普通障碍物，如果路径穿过边界定义的矩形区域就算碰撞
        for (ox, oy, w, h) in self.obs_boundary:
             # 边界扩展后的顶点 (这里即时计算，也可以预计算)
             delta = self.delta
             v1 = [ox - delta, oy - delta]
             v2 = [ox + w + delta, oy - delta]
             v3 = [ox + w + delta, oy + h + delta]
             v4 = [ox - delta, oy + h + delta]
             # AABB 快速排斥
             if max(start[0], end[0]) >= v1[0] \
                and v2[0] >= min(start[0], end[0]) \
                and max(start[1], end[1]) >= v1[1] \
                and v3[1] >= min(start[1], end[1]):
                 # 精确相交检查 (与四条边)
                 if self.is_intersect_line(start, end, v1, v2) or \
                    self.is_intersect_line(start, end, v2, v3) or \
                    self.is_intersect_line(start, end, v3, v4) or \
                    self.is_intersect_line(start, end, v4, v1):
                     # 需要小心处理正好沿着边界移动的情况
                     # 如果允许贴着边界走，这里的逻辑需要调整
                     # 当前逻辑：穿越边界区域就算碰撞
                     return True


        return False # 没有发生碰撞