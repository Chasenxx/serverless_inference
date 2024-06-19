from bayes_opt import BayesianOptimization
from get_latency import get_latency
log = print

def get_inference_time(b, c, m):
    # 输入数据四舍五入整数化，防止输入数据有问题
    b = round(b)
    c = round(c)
    m = round(m)

    # fake function
    # t = b ** 2 + c ** 2 + m + 1

    # real function
    try:
        t = get_latency(b, c, m)
    except:
        t = 60000
        # 出错时返回一个很大的数值  
    log('bs: {}, con: {}, mem: {}, latency: {}'.format(b, c, m, t))
    return t


def find_best_config():

    pbounds = {'b': (1, 16), 'c': (1, 3), 'm': (256, 10240)}

    optimizer = BayesianOptimization(
        f=get_inference_time,
        pbounds=pbounds,
        random_state=1,
        allow_duplicate_points=True,   # 离散贝叶斯优化, 开启允许重复点
        verbose=2,                     # 0--过程无输出, 1--输出找到的更加的解, 2--输出全部
        rate=10,
        SLO=200000
    )

    optimizer.cost_efficient(
        init_points=25,
        n_iter=15,   
    )

    min_cost = optimizer.cost_min
    # print(min_cost)
    return min_cost


def test():
    get_inference_time(7, 3, 1786)

if __name__ == "__main__":
    log(find_best_config())
    # test()
