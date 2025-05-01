"""
Class for implementing rrt* algorithm
@author: Yulin Li
@email: yline@connect.ust.hk
"""
# rrt_star_zh.py
import matplotlib.pyplot as plt
import numpy as np
import math
import random  # 引入随机库用于采样
from env_zh import Env # 导入包含中文注释的环境文件

class Node:
    """ RRT* 树中的节点 """
    def __init__(self, n, cost=0.0, parent=None):
        """
        初始化节点。
        :param n: 节点坐标 (x, y)
        :param cost: 从起点到此节点的代价（路径长度）
        :param parent: 父节点 (Node 对象)
        """
        self.x = n[0]  # x坐标
        self.y = n[1]  # y坐标
        self.cost = cost  # 代价
        self.parent = parent # 父节点

class RrtStar:
    """ RRT* 路径规划算法 """
    def __init__(self, xI, xG, step_len, search_radius, iter_max, env):
        """
        初始化 RRT* 规划器。
        :param xI: 起点坐标 (x, y)
        :param xG: 终点坐标 (x, y)
        :param step_len: 每步扩展的最大步长
        :param search_radius: RRT* 中用于寻找邻近节点和重接线的搜索半径
        :param iter_max: 最大迭代次数
        :param env: 环境对象 (Env)
        """
        self.tree = []  # 存储树中所有节点的列表
        self.path = []  # 存储最终找到的路径坐标 [(x,y), ...]
        self.node_start = Node(xI, 0.0)  # 起始节点，代价为0
        self.node_end = Node(xG, 0.0)    # 目标节点（初始代价为0，后面会更新）
        self.step_len = step_len          # 步长
        self.search_radius = search_radius # 邻域搜索半径 (RRT* 核心参数)
        self.iter_max = iter_max          # 最大迭代次数
        self.tree.append(self.node_start) # 将起点加入树中
        self.x_range = env.x_range        # 环境 x 范围
        self.y_range = env.y_range        # 环境 y 范围
        self.env = env                    # 环境对象
        # self.flag = 0 # 原始代码中的标记，此实现中未使用

    @staticmethod
    def distance(node1, node2):
        """计算两个节点之间的欧氏距离。"""
        return math.hypot(node2.x - node1.x, node2.y - node1.y)

    @staticmethod
    def get_dist_angle(start, end):
        """计算两个节点间的距离和角度。"""
        dist = RrtStar.distance(start, end)
        theta = math.atan2(end.y - start.y, end.x - start.x) # 计算方位角
        return dist, theta

    def get_random_node(self, goal_sample_rate=0.1):
        """
        在环境中随机采样一个点，有一定概率直接采样目标点。
        :param goal_sample_rate: 采样目标点的概率
        :return: 采样到的节点 (Node 对象)
        """
        # 以 goal_sample_rate 的概率直接采样目标点，引导搜索向目标前进
        if random.random() < goal_sample_rate:
            return self.node_end
        else:
            # 在环境范围内随机采样
            rand_x = random.uniform(self.x_range[0], self.x_range[1])
            rand_y = random.uniform(self.y_range[0], self.y_range[1])
            return Node((rand_x, rand_y)) # 返回一个临时 Node 对象

    def nearest_neighbor(self, rand_node_):
        """
        在当前树中找到距离给定随机节点最近的节点。
        :param rand_node_: 随机采样的节点
        :return: 树中最近的节点 (Node 对象)
        """
        # 使用 min 函数和 lambda 表达式高效查找最近邻
        nearest_node = min(self.tree, key=lambda node: self.distance(node, rand_node_))
        return nearest_node

    def new_state(self, start_node, end_node):
        """
        从 start_node 向 end_node 方向扩展 step_len 距离，生成新节点的状态（坐标）。
        :param start_node: 扩展的起始节点
        :param end_node: 扩展的目标方向节点（通常是随机采样点）
        :return: 新节点的状态（坐标），以 Node 对象形式返回（不含cost和parent）
        """
        dist, theta = self.get_dist_angle(start_node, end_node)

        # 如果目标距离小于步长，则新节点位置就是目标位置
        if dist <= self.step_len:
            new_x = end_node.x
            new_y = end_node.y
            return Node((new_x, new_y)) # 返回临时 Node
        else:
            # 否则，沿方向扩展 step_len 距离
            new_x = start_node.x + self.step_len * math.cos(theta)
            new_y = start_node.y + self.step_len * math.sin(theta)
            return Node((new_x, new_y)) # 返回临时 Node

    def find_near_neighbors(self, new_node_):
        """
        (RRT* 特有) 找到树中 new_node_ 指定搜索半径内的所有邻近节点。
        :param new_node_: 新生成的节点（或其状态）
        :return: 邻近节点列表 [Node, ...]
        """
        nnode = len(self.tree) + 1 # 树中节点数量（近似）
        # RRT* 理论上的最优搜索半径计算，可以根据实际情况调整或使用固定值
        # r = min(self.search_radius * math.sqrt((math.log(nnode) / nnode)), self.step_len * 2)
        r = self.search_radius # 直接使用参数指定的搜索半径

        # 列表推导式找出所有距离小于等于 r 的节点
        near_nodes = [
            node for node in self.tree if self.distance(node, new_node_) <= r
        ]
        return near_nodes

    def choose_parent(self, near_nodes, new_node_potential):
        """
        (RRT* 特有) 从邻近节点中为新节点选择最优父节点（代价最小且路径无碰撞）。
        :param near_nodes: 邻近节点列表
        :param new_node_potential: 新节点的潜在状态 (坐标)
        :return: (最佳父节点, 到达新节点的最小代价)，如果找不到有效父节点则返回 (None, float('inf'))
        """
        best_parent = None         # 初始化最佳父节点
        min_cost = float('inf')    # 初始化最小代价为无穷大

        if not near_nodes: # 安全检查，邻近节点列表不应为空（至少包含 nearest_node）
             return best_parent, min_cost

        # 遍历所有邻近节点
        for near_node in near_nodes:
            # 计算通过此邻近节点到达新节点的代价
            dist = self.distance(near_node, new_node_potential)
            potential_cost = near_node.cost + dist # cost(父) + dist(父->子)

            # 先检查路径是否碰撞，再比较代价
            if not self.env.is_collision(near_node, new_node_potential):
                # 如果路径无碰撞且代价更低
                if potential_cost < min_cost:
                    min_cost = potential_cost # 更新最小代价
                    best_parent = near_node   # 更新最佳父节点

        return best_parent, min_cost

    def rewire(self, new_node_, near_nodes_):
        """
        (RRT* 特有) 对新节点附近的邻近节点进行重接线操作。
        检查通过 new_node_ 到达这些邻近节点是否比它们原来的路径代价更低。
        如果是，则更新它们的父节点为 new_node_ 并更新代价。
        :param new_node_: 刚刚添加到树中的新节点
        :param near_nodes_: new_node_ 的邻近节点列表
        """
        if new_node_.parent is None: # 如果新节点本身就没有有效的父节点，则无法进行重连
            return

        # 遍历所有邻近节点
        for near_node in near_nodes_:
            # 不需要重连接 new_node_ 的直接父节点
            if near_node == new_node_.parent:
                continue

            # 计算通过 new_node_ 到达 near_node 的代价
            dist_new_to_near = self.distance(new_node_, near_node)
            potential_cost = new_node_.cost + dist_new_to_near # cost(新) + dist(新->邻)

            # 检查这条新路径是否比 near_node 原来的路径代价更低
            if potential_cost < near_node.cost:
                # 检查新路径 (new_node_ -> near_node) 是否无碰撞
                if not self.env.is_collision(new_node_, near_node):
                    # 重接线：更新 near_node 的父节点和代价
                    near_node.parent = new_node_
                    near_node.cost = potential_cost
                    # 注意：标准的 RRT* 通常只更新被重连接节点的代价。
                    # 如果其子节点的代价也需要更新，则需要进行成本传播，会更复杂。

    def extract_path(self, goal_node):
        """
        从目标节点回溯，提取从起点到终点的路径。
        :param goal_node: 已连接到树的目标节点 (包含父节点信息)
        :return: 路径坐标列表 [(x,y), ...], 从起点到终点。若失败则返回空列表。
        """
        path_nodes = []           # 存储路径节点的坐标
        current_node = goal_node  # 从目标节点开始回溯

        # 循环回溯，直到找到起点（parent 为 None）
        while current_node is not None:
            path_nodes.append((current_node.x, current_node.y)) # 添加当前节点坐标
            current_node = current_node.parent                  # 移动到父节点

        # 检查回溯的终点是否确实是起点
        # (理论上 goal_node 必须能回溯到 start_node，除非 find_goal_parent 逻辑有误)
        if path_nodes[-1] != (self.node_start.x, self.node_start.y):
             print("路径提取失败 - 目标节点未能连接回起点?")
             return [] # 提取失败返回空列表

        # 列表是反的（从终点到起点），需要反转
        return path_nodes[::-1]

    def find_goal_parent(self):
       """
       在所有迭代结束后，检查树中是否有节点能以低成本、无碰撞地连接到最终目标点。
       将目标节点连接到最优的父节点上。
       :return: 连接后的目标节点 (Node 对象)，如果无法连接则返回 None。
       """
       near_goal_nodes = []             # 存储可以连接到目标的候选父节点
       min_cost_to_goal = float('inf')  # 到达目标的最小总代价
       best_parent_for_goal = None      # 目标的最佳父节点

       # 遍历树中所有节点，寻找可能的父节点
       for node in self.tree:
           dist_to_goal = self.distance(node, self.node_end) # 计算节点到目标的距离

           # 检查连接是否可行（距离足够近）且无碰撞
           # 这里允许最终连接步长稍大于 step_len，增加成功连接的概率
           if dist_to_goal <= self.step_len * 1.5 :
               if not self.env.is_collision(node, self.node_end): # 检查 node -> goal 是否碰撞
                   # 计算通过此 node 到达 goal 的总代价
                   cost_via_node = node.cost + dist_to_goal
                   # 如果找到更优的路径
                   if cost_via_node < min_cost_to_goal:
                       min_cost_to_goal = cost_via_node    # 更新最小代价
                       best_parent_for_goal = node         # 更新最佳父节点

       # 如果找到了有效的父节点
       if best_parent_for_goal:
           self.node_end.cost = min_cost_to_goal      # 设置目标节点的最终代价
           self.node_end.parent = best_parent_for_goal # 设置目标节点的父节点
           return self.node_end # 返回更新后的目标节点
       else:
           print("目标节点无法连接到 RRT* 树。")
           return None # 无法连接，返回 None

    def rrt_star_planning(self, animation=False):
        """RRT* 规划算法主函数。"""
        # 主循环，迭代 iter_max 次
        for i in range(self.iter_max):
            # 1. 随机采样节点 (带目标偏置)
            rand_node = self.get_random_node()

            # 2. 找到树中离采样点最近的节点 (Nearest Neighbor)
            nearest_node = self.nearest_neighbor(rand_node)

            # 3. 从最近邻节点向随机节点方向扩展，生成新节点状态 (Steer)
            new_node_potential = self.new_state(nearest_node, rand_node)

            # 4. 检查新节点状态本身是否在障碍物内 (可选，但可提早排除无效状态)
            #    env.is_collision 会检查端点，所以理论上这里可以省略
            #    但显式检查可以避免后续不必要的计算
            if self.env.is_inside_obs(new_node_potential):
                 # print(f"迭代 {i}: 新节点状态在障碍物内。")
                 continue # 跳过本次迭代

            # === RRT* 核心步骤 ===
            # 5. 寻找新节点状态附近的邻近节点 (Find Near Neighbors)
            near_nodes = self.find_near_neighbors(new_node_potential)
            if not near_nodes: # 安全检查，至少包含 nearest_node (除非树为空)
                 near_nodes = [nearest_node] # 理论上不会执行到这里

            # 6. 选择父节点 (Choose Parent)：从邻近节点中选择代价最小且无碰撞的父节点
            parent_node, cost_to_new = self.choose_parent(near_nodes, new_node_potential)

            # 7. 如果找到了有效的父节点（路径无碰撞）
            if parent_node is not None:
                # 创建实际的新节点，并设置其代价和父节点
                new_node = Node((new_node_potential.x, new_node_potential.y),
                                cost=cost_to_new,
                                parent=parent_node)
                self.tree.append(new_node) # 将新节点添加到树中

                # 8. 重接线 (Rewire)：优化新节点邻域内的路径
                self.rewire(new_node, near_nodes) # 注意传入的是刚添加的 new_node
            else:
                 # print(f"迭代 {i}: 未找到新节点的有效父节点。")
                 continue # 无法连接，跳过本次迭代

            # (可选) 打印迭代进度
            if i % 100 == 0:
                print(f"迭代: {i}/{self.iter_max}, 树大小: {len(self.tree)}")

            # (可选) 提前检查是否能连接到目标点 (可能加速收敛，但 RRT* 通常跑满迭代次数以获得更优解)
            # dist_to_goal = self.distance(new_node, self.node_end)
            # if parent_node and dist_to_goal <= self.step_len and not self.env.is_collision(new_node, self.node_end):
            #     potential_goal_cost = new_node.cost + dist_to_goal
            #     current_goal_cost = getattr(self.node_end.parent, 'cost', float('inf')) + self.distance(self.node_end.parent, self.node_end) if self.node_end.parent else float('inf')
            #     if potential_goal_cost < current_goal_cost:
            #          self.node_end.parent = new_node
            #          self.node_end.cost = potential_goal_cost
            #          print(f"迭代 {i}: 找到一条到目标的路径!")
                     # break # 可以考虑找到路径后提前终止

        # === 迭代结束后提取路径 ===
        print("达到最大迭代次数。尝试寻找最终路径...")
        final_goal_node = self.find_goal_parent() # 尝试连接目标节点

        # 如果成功连接到目标
        if final_goal_node:
            self.path = self.extract_path(final_goal_node) # 回溯提取路径
            if self.path: # 如果路径提取成功
                 print(f"找到路径！ 长度: {final_goal_node.cost:.2f}, 节点数: {len(self.path)}")
            else: # 路径提取失败（理论上不应发生）
                 print("找到目标父节点后，路径提取失败。")
                 self.path = [] # 保证 path 是空列表
        else: # 未能连接到目标
            print("未能建立到目标节点的路径。")
            self.path = [] # 保证 path 是空列表

        # === 绘图 ===
        self.env.plot_grid("RRT* Planning Result") # 绘制环境、起点、终点
        self.env.plot_visited(self.tree, animation=animation) # 绘制 RRT* 树 (绿色)
        self.env.plot_path(self.path) # 绘制最终路径 (红色)
        plt.show() # 显示绘图窗口

if __name__ == '__main__':
    # 起点和终点坐标 (根据图片设置)
    x_start = (18, 8)  # 起点 (蓝色方块)
    x_goal = (37, 18)  # 终点 (绿色方块)

    # --- RRT* 参数 ---
    STEP_LEN = 3.0       # 每步最大扩展长度
    SEARCH_RADIUS = 10.0 # 邻域搜索半径 (RRT*) - 可调整影响性能和路径质量
    ITER_MAX = 2000      # 最大迭代次数
    # ---

    # 创建环境对象
    env_ = Env(x_start, x_goal, "RRT* Env") # 使用带中文注释的 Env 类

    # (可选) 在运行 RRT* 前显示地图设置
    # env_.plot_grid("地图设置")
    # plt.show()
    # exit() # 如果只想看地图，取消此行注释

    # 创建 RRT* 规划器对象
    rrt_star = RrtStar(x_start, x_goal, STEP_LEN, SEARCH_RADIUS, ITER_MAX, env_)
    # 执行规划
    rrt_star.rrt_star_planning(animation=False) # animation=True 可观看树生长动画
    