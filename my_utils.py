import functools
import time, base64
import boto3
log = print

def program_run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t1 = time.perf_counter()
        res = func(*args, **kw)
        t2 = time.perf_counter()
        print('####### This Program run: {} ms #######'.format((t2 - t1) * 1000))
        return res
    return wrapper


# 读取图片源数据-->base64编码-->str
# input: filename
# output: str with base64
def file2str(filename):
    data_str = ''
    with open(filename, 'rb') as f:
        raw_data = f.read()
        data_str = base64.b64encode(raw_data)
        data_str = data_str.decode()
    return data_str


def get_lambda_url(functionName):
    client = boto3.client('lambda')
    response = client.get_function_url_config(
        FunctionName=functionName
    )
    url = response['FunctionUrl']
    log('Funciton url is: ', url)
    return url

