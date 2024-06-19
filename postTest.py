from time import sleep
from my_utils import program_run_time, file2str, get_lambda_url
from threading import Thread
import requests
import json
import queue
log = print


# 全局变量
proxyQ = queue.Queue()
CONC = 2
BS = 6
LENGTH = 8
TIMEOUT = 2000  # ms


# 从指定图片生成 json 字符串
# input: image file name list
# output: json string
def gen_post_json(filenames: list, concurrency, batchsize, size, raw_data=False):
    file_datas = []

    if raw_data:
        file_datas = filenames
    else:
        file_datas = [file2str(f) for f in filenames]

    batch_data = {
        "concurrency": concurrency,
        "batchsize": batchsize,
        "size": size,
        "file-type": "image",
        "data": file_datas,
        "isBase64Encoded": True
    }

    res = json.dumps(batch_data)
    return res


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

    log(r.status_code)
    log(r.json())
    return r.json()


# 从队列中获取数据，发送请求
def send_from_queue():
    length = proxyQ.qsize()
    if length > CONC * BS:
        length = CONC * BS
    
    data = []
    for i in range(length):
        data.append(proxyQ.get())
    # send
    functionName = "Ort-resnet50"
    url = get_lambda_url(functionName)
    send_batch(url, data, concurrency=CONC, batchsize=BS, size=length, raw_data=True)


# 检测队列长度和超时
def proxy():
    while True:
        if proxyQ.qsize() >= CONC * BS:
            # queue full
            log('send package-------->')
            # TODO send in new threading
            send_from_queue()


# send 用于直接发送一个请求
def send():
    filename = './mini_batch/1.jpg'
    data = file2str(filename)
    proxyQ.put(data)
    log('size in Queue is:', proxyQ.qsize())
    log('send----->')


# 固定 RPS 的请求测试器
# rps=10 代表每秒 10 个请求
def fake_request(rps=4):
    # 间隔 interval 发送一个请求
    interval = 1 / rps
    while True:
        send()
        sleep(interval)


def main():
    # 请求生成进程
    p_generator = Thread(target=fake_request, args=())
    p_generator.start()

    # 队列处理进程
    p_proxy = Thread(target=proxy, args=())
    p_proxy.start()


def test():
    functionName = "GetLatency"
    url = get_lambda_url(functionName)

    filenames = ['./mini_batch/1.jpg', './mini_batch/2.jpg', './mini_batch/3.jpg', './mini_batch/4.jpg',
                 './mini_batch/5.jpg', './mini_batch/6.jpg', './mini_batch/7.jpg', './mini_batch/7.jpg', ]
    send_batch(url, filenames, 2, 4, 8, raw_data=False)


if __name__ == "__main__":
    # main()
    test()