from time import sleep
from my_utils import program_run_time, file2str, get_lambda_url
from threading import Thread, Lock
import requests, json, queue, time
from get_latency import update_memory
import numpy as np
from get_latency import get_latency as objective
from my_bo import MyBayesOpt
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from bayesian_changepoint_detection import offline_likelihoods as offll
log = print


file_lock = Lock()

# 发送batch图片给 Lambda-Function-url
# input: filenames
# @program_run_time
def send_batch(url, filenames: list, concurrency, batchsize, size, raw_data=False):

    file_datas = []
    if raw_data:
        file_datas = filenames
    else:
        file_datas = [file2str(f) for f in filenames]

    # send package format
    batch_data = {
        "concurrency": concurrency,
        "batchsize": batchsize,
        "size": size,
        "file-type": "image",
        "data": file_datas,
        "isBase64Encoded": True
    }

    # send across HTTP
    json_data = json.dumps(batch_data)
    headers = {"content-type": "application/json"}
    r = requests.post(url, headers=headers, data=json_data)

    # log(r.status_code)
    # log(r.json())
    # return r.json()


def send_batch_isolated(url, filenames: list, concurrency, batchsize, size, raw_data=False):
    t = Thread(target=send_batch, args=(url, filenames, concurrency, batchsize, size, raw_data))
    t.start()


def test():
    functionName = "Ort-resnet50"
    url = get_lambda_url(functionName)

    filenames = ['./mini_batch/1.jpg', './mini_batch/2.jpg', './mini_batch/3.jpg', './mini_batch/4.jpg',
                 './mini_batch/5.jpg', './mini_batch/6.jpg', './mini_batch/7.jpg', './mini_batch/7.jpg', ]
    
    send_batch_isolated(url, filenames, 2, 4, 8, raw_data=False)


class MyServerless(Thread):
    def __init__(self, url, filenames, timestamps, concurrency, batchsize, size, raw_data=False):
        Thread.__init__(self)
        self.timestamps = timestamps
        self.concurrency = concurrency
        self.batchsize = batchsize
        self.size = size
        self.raw_data = raw_data
        self.url = url
        self.filenames = filenames

    def run(self):
        start_time = time.perf_counter()
        # send request
        send_batch(self.url, self.filenames, self.concurrency, self.batchsize, self.size, self.raw_data)

        end_time = time.perf_counter()
        duration = end_time - start_time
        last_timestamp = self.timestamps[-1]
        end_timestanp = last_timestamp + duration

        # add data to file
        global file_lock
        file_lock.acquire()
        with open('request_log.txt', 'a') as f:
            for ts in self.timestamps:
                data = f'{ts}, {end_timestanp}\n'
                f.write(data)

        with open('actual_count.txt', 'a') as f2:
            f2.write(f'{self.size}\n')

        file_lock.release()


class VirtualQueue:
    def __init__(self):
        self.BatchSize = 16
        self.Concurrency = 1
        self.Memory = 1024
        self.SLO = 2
        self.Timeout = 0.2
        self.R_status = 0
        self.interval_pointer = 0
        self.rps_pointer = 0
        self.timestamp = 0
        self.window_size = 5 * 60
        self.functionNameOrigin = "Ort-resnet50"
        self.functionName = self.functionNameOrigin
        self.url = get_lambda_url(self.functionName)   # 服务的 url
        self.memory_status = 1

        # self.requests = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        # self.interval = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # self.loadInterval()
        self.loadTrace()
        self.rps = []

    def loadInterval(self, fileName='workload/workload-2.json'):
        with open(fileName, 'r') as f:
            self.interval = json.load(f)

    def loadTrace(self, filename='workload/twitter-trace/TW_may25_0-1'):
        with open(filename, 'r') as f:
            self.interval = [float(d) for d in f.readlines()]

    def send(self):
        length = self.Concurrency * self.BatchSize
        request_timestamp = []
        wait_time = 0
        count = 0

        for i in range(length):
            current_interval = self.interval[self.interval_pointer]
            if wait_time + current_interval <= self.Timeout or wait_time == 0:
                wait_time += current_interval
                self.interval_pointer += 1
                request_timestamp.append(self.timestamp)
                count += 1
                self.timestamp += current_interval
            else:
                break
        
        log(f'Prepare <{self.Concurrency}>*<{self.BatchSize}> requests')
        log(f'Actual send <{count}> requests')
        log(f'wait_time/timeout = <{wait_time}>/<{self.Timeout}>')    
        time.sleep(wait_time)

        # send batch in new thread

        filenames = ['./mini_batch/1.jpg'] * count
        # send_batch_isolated(self.url, filenames, self.Concurrency, self.BatchSize, count, raw_data=False)
        instance = MyServerless(self.url, filenames, request_timestamp, self.Concurrency,self.BatchSize, count)
        instance.start()
    

    def freshRPS(self):
        while (self.rps_pointer <= self.interval_pointer):
            interval = self.interval[self.rps_pointer]
            if interval == 0:
                # 间隔为 0, rps 设置和前面一样
                self.rps.append(self.rps[-1])
            else:
                self.rps.append(1 / interval)
            self.rps_pointer += 1


    def change_point_detection(self):
        step = 5

        # 初始化用于保存最后一个变点后的序列平均值
        average_values = []

        # 循环处理，模拟每五分钟运行一次变点检测
        current_window = []
        current_count = 0
        tmp_pointer = self.interval_pointer

        while current_count + self.interval[tmp_pointer] < self.window_size and tmp_pointer > 0:
            current_window.append(self.interval[self.tmp_pointer])
            tmp_pointer -= 1

        # 贝叶斯变点检测
        Q, P, Pcp = offcd.offline_changepoint_detection(
            current_window,
            prior_func=offcd.const_prior(len(current_window)),
            ll_func=offll.exponential_obs_log_likelihood,
            truncate=-40  # 使用更高的截断值确保变点在整个序列中
        )
        
        # 找到变点的索引
        change_points = np.where(np.exp(Pcp).sum(0) > 0.5)[0]  # 这里设定了一个阈值 0.5
        
        # 计算最后一个变点后的序列的平均值
        if len(change_points) > 0:
            last_changepoint_index = change_points[-1]
            sequence_after_last_cp = current_window[last_changepoint_index + 1:]
            if len(sequence_after_last_cp) > 0:
                average_value = np.mean(sequence_after_last_cp)
            else:
                average_value = None  # 如果没有数据点，设置为 None
        else:
            average_value = None  # 如果没有变点，设置为 None
        
        if average_value is None:
            return average_value
        else:
            new_rps_status = int(1.0 / average_value) % step
            return new_rps_status
    

    def updateConfig(self, b, c, m, timeout, rps_status):
        self.BatchSize = b
        self.Concurrency = c
        self.Memory = m
        self.Timeout = timeout
        self.R_status = rps_status

        # update serverless memory
        if self.memory_status == 1:
            functionName = f"{self.functionNameOrigin}-1"
        else:
            functionName = f"{self.functionNameOrigin}-2"
        update_memory(functionName, m)

        # wait for change memory
        time.sleep(5)

        # virtual request
        req_count = (self.SLO - self.Timeout) * self.R_status / (self.BatchSize * self.Concurrency)
        
        concurrency = 1
        batchsize = 1
        count = 1
        filenames = ['./mini_batch/1.jpg'] * count
        request_timestamp = [0]
        for _ in range(req_count):
            instance = MyServerless(self.url, filenames, request_timestamp, concurrency, batchsize, count)
            instance.start()

        self.functionName = functionName
        self.url = get_lambda_url(self.functionName)
    

    def get_new_config(self, rps, slo):
        rps = 10
        slo = 2000

        bo = MyBayesOpt(objective, rps, slo, 16, 2, 128, 3072)
        bo.optimize(30)
        result = bo.get_result()
        # result = [b, c, m, delay]
        delay = result[3]
        timeout = slo - delay
        result[3] = timeout
        return result


    def freshConfig(self):
        # 新进程中执行, 并且定时执行
        
        # 1. 运行贝叶斯变点检测，判断当前rps是否和设定的rps在同一个状态
        # 2. 如果一样，则不更新，否则更新
        # 3. 更新后，将新的rps和对应的配置文件写入json文件中
        new_rps_status = self.change_point_detection()
        if new_rps_status != self.R_status:
            # 每个 slo 对应一个记录状态和响应配置的文件
            filepath = f'./slo_{self.SLO}_config.json'
            config = json.load(open(filepath, 'r'))
            # example config
            # config = {
            #     '3': [8, 2, 2048, 3000],
            #     '4': [8, 2, 2048, 3000],
            #     '5': [8, 2, 2048, 3000]
            # }
            if str(new_rps_status) in config:
                # 查表找到对应方案并且更新
                tmp = config[str(new_rps_status)]
                self.updateConfig(tmp[0], tmp[1], tmp[2], tmp[3], rps_status=new_rps_status)
            else:
                # BO 求解新方案
                new_config = self.get_new_config(self.rps, self.SLO)
                self.updateConfig(new_config[0], new_config[1], new_config[2], new_config[3], rps_status=new_rps_status)
                
                # 更新 json 文件
                config[str(new_rps_status)] = new_config
                json.dump(config, open(filepath, 'w'))

                log('add new config:: ', new_config)


    def periodic_execution(self):
        while True:
            self.freshConfig()
            time.sleep(self.window_size)  # 等待5分钟 (300秒)


    def run(self):
        while True and (self.interval_pointer + self.BatchSize * self.Concurrency < len(self.interval)):
            self.send()
            # self.freshRPS()

            # 子线程定时执行配置更新
            thread = threading.Thread(target=self.periodic_execution)
            thread.daemon = True
            thread.start() 
        

def test_virtual_queue():
    vq = VirtualQueue()
    # vq.send()
    vq.run()


def test_myserverless():
    filenames = ['./mini_batch/1.jpg'] * 8
    instance = MyServerless(url=get_lambda_url('Ort-resnet50'), filenames=filenames,
                            timestamps=[0,1,2,3,4,5,6,7], concurrency=1, batchsize=8, size=8)
    instance.start()


if __name__ == "__main__":
    # main()
    # test()
    test_virtual_queue()
    # test_myserverless()