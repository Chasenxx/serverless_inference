import numpy as np
from math import ceil, exp
from scipy.stats import norm
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
log = print


class MyBayesOpt(object):
    def __init__(self, f, R, SLO, B_max, C_max, M_down, M_up):
        self.SLO = SLO # ms
        self.time_slot = 300  # s
        self.objective = f
        self.model = GaussianProcessRegressor()
        self.X = np.empty([0,3], dtype=float)
        self.delay = np.asarray([])
        self.cost = np.asarray([])
        self.count = 0
        self.R = R

        # 最优解
        self.min_cost = np.inf
        self.min_cost_index = 0

        # 搜索界限
        self.B_max = B_max    # b =[1 .. B_max]
        self.C_max = C_max    # c =[1 .. C_max]
        self.M_down = M_down  # m =[M_down .. M_up]
        self.M_up = M_up


    def cal_cost(self, b, c, m, delay):
        cost = m * ceil((self.R * self.time_slot) / (b * c)) * delay
        return cost
    
    # 添加样本
    def add_sample(self, b, c, m):
        try:
            y = self.objective(b, c, m)
        except:
            log("error in get latency")
            return
        
        # 归一化输入参数
        self.X = np.append(self.X, [[b/self.B_max, c/self.C_max, m/self.M_up]], axis=0)

        # 归一化延迟
        cost = self.cal_cost(b, c, m, y)
        self.cost = np.append(self.cost, cost)
        self.delay = np.append(self.delay, y/400)

        # 超时的方案不能用于更新最小成本
        if (y > self.SLO - (b*c)/self.R):
            log("out of SLO")
        elif cost < self.min_cost:
            self.min_cost = cost
            self.min_cost_index = self.count
            log("new optimal found")
        
        self.count += 1
        log(f'add point: b={b}, c={c}, m={m}, latency={y}, cost={cost}')

    # 显示已采样点
    def show_space(self):
        print("count of sample is:", self.count)
        print("X is: ")
        print(np.multiply(self.X, [self.B_max, self.C_max, self.M_up]))
        print("y is: ")
        print(np.multiply(self.delay, 400))
        print("cost is: ")
        print(self.cost)
        print("optimal conf is: ", self.get_min_conf())
        print("return result: ", self.get_result())


    # 拟合模型
    def fit(self):
        self.model.fit(self.X, self.delay)

    # 预测数据
    def predict(self, x_new):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.model.predict(x_new, return_std=True)
            return mean*400, std*400

    # 返回最优配置
    def get_min_conf(self):
        conf = self.X[self.min_cost_index]
        return np.multiply(conf, [self.B_max, self.C_max, self.M_up]).tolist()
    
    # 返回优化结果
    def get_result(self):
        conf = self.X[self.min_cost_index]
        process_time = self.delay[self.min_cost_index]
        conf = np.multiply(conf, [self.B_max, self.C_max, self.M_up]).tolist()
        process_time = np.multiply(process_time, [400]).tolist()
        conf.append(process_time[0])
        return conf

    
    # SLO 约束函数
    def slo_constrain(self, b, c, latency):
        threshold = self.SLO - (b * c)/self.R
        if latency < threshold:
            return 1.0
        else:
            return exp((threshold - latency) * 100 / threshold)

    # 采集函数
    def EI(self, b, c, m, mean, std, c_min, xi=0.005):
        alpha = m * ceil((self.R * self.time_slot) / (b * c))     # 成本系数 
        mean_new = alpha * mean                                   # 新的均值
        std_new = alpha * std                                     # 新的标准差
  
        a = c_min - mean_new - xi
        z = a / std_new
        ei = a * norm.cdf(z) + std_new * norm.pdf(z)

        # SLO 约束比例系数
        ei = ei * self.slo_constrain(b, c, mean)
        return ei

    # 每个点都来求一下 EI，找出最大的那个
    def find_next_sample(self):
        max_ei = -np.inf
        next_sample = []
        # b,c 每一个都遍历, m 每隔 32 遍历一次
        for b in range(1, self.B_max+1):
            for c in range(1, self.C_max+1):
                for m in range(self.M_down, self.M_up+1, 32):
                    if [b/self.B_max, c/self.C_max, m/400] in self.X.tolist():
                        continue
                    mean, std = self.predict(np.asarray([[b/self.B_max, c/self.C_max, m/self.M_up]]))
                    ei = self.EI(b, c, m, mean, std, self.min_cost)
                    if ei > max_ei:
                        max_ei = ei
                        next_sample = [b, c, m]
        return next_sample


    def optimize(self, n_iter=10):
        init_b = [1, 2, 4, 8, 16]
        init_c = [1, 2]
        init_m = [128, 256, 512, 1024, 2048, 3072]

        for m in init_m:
            for b in init_b:
                for c in init_c:
                    self.add_sample(b, c, m)
        self.fit()

        # 迭代优化
        for i in range(n_iter):
            next_sample = self.find_next_sample()
            b = next_sample[0]
            c = next_sample[1]
            m = next_sample[2]

            self.add_sample(b, c, m)
            self.fit()
            log("Next sample is: ", next_sample)
            log("Optimal cost is: ", self.min_cost)
            log("Optimal conf is: ", self.get_min_conf())


def test_BO():

    # 目标函数
    from get_latency import get_latency as objective
    bo = MyBayesOpt(objective, 10, 2000, 16, 2, 1024, 10240)
    bo.optimize(30)
    bo.show_space()


if __name__ == '__main__':
    # test_BO()
    # test_find_min()
    # draw_3d()
    test_BO()
