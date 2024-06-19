import boto3
from time import sleep, perf_counter
from postTest import send_batch
from my_utils import get_lambda_url, program_run_time
log = print


# 判断 Serverless 函数是否存在
def is_exist(funName):
    try:
        client = boto3.client('lambda')
        client.get_function(FunctionName=funName)
        return True
    except Exception:
        return False


def update_memory(funName, memorySize):
    client = boto3.client('lambda')
    client.update_function_configuration(
        FunctionName=funName,
        MemorySize=memorySize
    )
    log('Update Lambda:[{}] to {} MB'.format(funName, memorySize))


def deploy_lambda(stackName='DLStack', funName='Ort-resnet50', roleName='DLRole'):
    # change template url
    # e.g. templateURL = 'https://dl-model-bucket-shen.s3.ap-southeast-1.amazonaws.com/DeepLearning_Serverless_CF.json'
    templateURL = ''
    client = boto3.client('cloudformation')
    client.create_stack(
        StackName=stackName,
        TemplateURL=templateURL,
        Parameters=[
            {
                'ParameterKey': 'ModelAndCodeBucket',
                'ParameterValue': 'dl-model-bucket-shen',
            },
            {
                'ParameterKey': 'DLBundleObjectKey',
                'ParameterValue': 'resnet50.zip',
            },
            {
                'ParameterKey': 'myFunctionName',
                'ParameterValue': funName,
            },
            {
                'ParameterKey': 'myRoleName',
                'ParameterValue': roleName,
            },
        ],
        Capabilities=[
            'CAPABILITY_NAMED_IAM'
        ],
        OnFailure='DELETE',
    )
    log('Create function: {}'.format(funName))


# 获取配置方案的推理延迟
# INPUT:  c --> concurrency
#         b --> batchsize
#         m --> memory size of Lambda
# OUTPUT: corresponding latency
# @program_run_time
def get_latency(b, c, m):

    # 先判断采样函数是否存在
    if not is_exist('GetLatency'):
        # Create GetLatnecy stack
        deploy_lambda('StackGetLatency', 'GetLatency', 'RoleGetLatency')
        log('Creat get latency stack')

        # 设置 65s 创建延迟
        log('waitting ... ')
        sleep(65)

    # Update memory first
    update_memory('GetLatency', m)
    sleep(2)

    # invoke function with c*b
    functionName = "GetLatency"
    url = get_lambda_url(functionName)

    request_num = b * c
    filenames = ['./mini_batch/1.jpg'] * request_num

    # 三次采样取后两次平均
    latency = []
    for _ in range(3):
        start = perf_counter()
        send_batch(url, filenames, batchsize=b, concurrency=c, size=request_num)
        end = perf_counter()
        duration = (end - start) * 1000
        latency.append(duration)
    
    latency_mean = (latency[1] + latency[2]) / 2
    # log('bs: {}, con: {}, mem: {}, latency array: {}, latency: {}'.format(b, c, m, latency, latency_mean))
    return latency_mean
    
    
def test():
    # deploy_lambda('DLtest', 'Ort-test', 'RoleTest')
    # update_memory('Ort-resnet50', 2048)
    # res = is_exist('GetLatency')
    # log(res)
    b = 3
    c = 1
    m = 9728
    latency = get_latency(b, c, m)
    log('latency is:', latency)


if __name__ == "__main__":
    test()

