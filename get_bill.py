import boto3
log = print


# input: logGroupName
# output: ['logStream1', 'logStream2']
def get_log_streams(logGroupName):
    # logGroupName = "/aws/lambda/{FuncitonName}"
    client = boto3.client('logs')
    response = client.describe_log_streams(
        logGroupName=logGroupName,
        orderBy='LastEventTime',
        descending=True
    )
    res = []
    streams = response["logStreams"]
    for s in streams:
        StreamName = s["logStreamName"]
        res.append(StreamName)
    return res


# input: logGroupName, logStreamName
# output: ['message1', 'message2']
def get_log_event(logGroupName, logStreamName):
    client = boto3.client('logs')
    response = client.get_log_events(
        logGroupName=logGroupName,
        logStreamName=logStreamName,
        startFromHead=True
    )
    res = []
    events = response["events"]
    for e in events:
        message = e["message"]
        res.append(message)
    return res


# input: message (['message1', 'message2'])
# output: [['duration1','memory1'], ['duration2','memory2']]
def get_bill_duration(message):
    res = []
    for m in message:
        if "Billed Duration" in m:
            print(m)
            bill_log = m.split('\t')
            # bill_log = ['RequestId', 'Duration', 'Billed_Duration', 'MemorySize', 'MaxMemoryUsed']
            bill_duration = bill_log[2]
            memory = bill_log[3]
            res.append([bill_duration, memory])
    return res


def calulate_bill(data):
    # ap-southeast-1 (Singapore)
    unit = 0.0000166667 #  GB*s / dollar
    bill_time = 0
    for d in data:
        duration = int(d[0].split(' ')[2])  # ms
        memory = int(d[1].split(' ')[2])    # MB
        tmp_bill_time = (duration / 1000) * (memory / 1024)  # s * GB
        # log(duration, memory)
        bill_time += tmp_bill_time
    return unit * bill_time
    

def get_lambda_bill(LambdaName):
    cost = 0
    logGroupName = '/aws/lambda/' + LambdaName
    streams = get_log_streams(logGroupName)
    for s in streams:
        message = get_log_event(logGroupName, s)
        bill_message = get_bill_duration(message)
        tmp_cost = calulate_bill(bill_message)
        cost += tmp_cost
    return cost


def main():
    LambdaName = 'Ort-resnet50'
    cost = get_lambda_bill(LambdaName)
    log('Cost of {} is: {}'.format(LambdaName, cost))


if __name__ == "__main__":
    main()

