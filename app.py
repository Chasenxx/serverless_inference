from math import ceil
from inference_onnx_resnet50 import cpu_ort_inference, gen_batch, cpu_ort_infer_batch
import json, base64, io
from imageio.v2 import imread
from math import ceil
from multiprocessing import  Process, Manager
log = print


def api_gateway(event, context):
    log("Input event is:", event) 

    requestContext = event["requestContext"]
    headers = event["headers"]
    http = requestContext["http"]
    method = http["method"]

    raw_image_data = ""

    return_data = {}

    if method == "GET":
        # Process get
        log("get")

        res = cpu_ort_inference('./mini_batch')
        res = res[0]
        res = ' '.join(res)

        # http data
        return_data = {
            'statusCode': 200,
            "headers": {
                "Content-Type": "text/plain;charset=utf-8",
            },
            'body': 'Hello from AWS Lambda using Python result is: \n{}'.format(res),

        }

    elif method == "POST":
        # Process post
        log("post")
        content_type = headers["content-type"]
        if content_type == "image/png":
            data = event["body"]
            if event["isBase64Encoded"]:
                raw_image_data = data

        return_data = {
            'statusCode': 200,
            "headers": {
                "Content-Type": "image/png",
            },
            'body': raw_image_data,
            'isBase64Encoded': 'true'
        }

    return return_data


def lambda_handler_old(event, context):
    # 判断是 test 还是 API-Gateway
    requestContext = event.get("requestContext", "test")
    if requestContext == "test":
        # 进入测试
        res = cpu_ort_inference('./mini_batch')
        res = res[0]
        res = ' '.join(res)
        return {
            'statusCode': 200,
            'body': 'Hello from Lambda! Result is: {}'.format(res)
        }
    else:
        # 进入API-Gateway
        return api_gateway(event, context)

    
# 从 post 中解析出图片
# input: json string
# output: ndarray list
def extract_from_json(json_input):
    res = []
    # concurrency = json_input["concurrency"]
    # batchsize = json_input["batchsize"]
    size = json_input["size"]
    raw_datas = json_input["data"]
    for i in range(size):
        file = base64.b64decode(raw_datas[i].encode())
        f = io.BytesIO(file)
        img = imread(f)
        res.append(img)
    return res


def lambda_handler(event: dict, context):

    # extract data
    flag = event.get('batchsize', 'API-Gateway')
    data = {}
    if flag == 'API-Gateway':
        # from api-gateway
        data = json.loads(event["body"])
    else:
        # from raw json
        data = event

    img_list = extract_from_json(data)
    concurrency = data["concurrency"]

    # prepare data 均匀分割
    # img_data = [[data1, data2], [data3, data4], [data5, data6]]
    # input = [tensor1, tensor2, tensor3]
    size = data["size"]
    remain = size % concurrency
    unit = size // concurrency
    front_count = remain * (unit + 1)

    front_data = [img_list[i: i + unit + 1] for i in range(0, front_count, unit + 1)]
    tail_data = [img_list[i: i + unit] for i in range(front_count, size, unit)]
    img_data = front_data + tail_data
    
    input = [gen_batch(i, is_raw=True) for i in img_data]
    
    # infer in multi process
    manager = Manager()
    result_dict = manager.dict()

    jobs = []
    for i in range(concurrency):
        p = Process(target=cpu_ort_infer_batch, args=(input[i], i, result_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    keys = result_dict.keys()
    values = result_dict.values()
    res = sorted(zip(keys, values))
    _, values = zip(*res)
    values = list(values)
    tmp = [item for sublist in values for item in sublist]
    log("predicted is:", tmp)

    response = {
        'concurrency': data['concurrency'],
        'batchsize': data['batchsize'],
        'inference': tmp
    }

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }


if __name__ == "__main__":
    from data import body
    event = json.loads(body)
    log(lambda_handler(event, {}))