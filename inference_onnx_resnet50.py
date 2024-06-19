import onnxruntime
import numpy as np
import os
from my_utils import program_run_time
from labels import imagenet_labels
from imageio.v2 import imread
import boto3
log = print


# download model from s3 bucket
# 从 s3 下载模型文件
MODEL_FILENAME = 'resnet50.onnx'
BUCKET_NAME = 'dl-model-bucket-shen'
LOCATION = '/tmp/resnet50.onnx'
# for test
# LOCATION = 'resnet50.onnx'
def download_model():
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET_NAME).download_file(MODEL_FILENAME, LOCATION)

download_model()

# input: array
# output: array with normalize
# Normalize ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# 对 array 进行标准归一化
def array_normalize(input):
    # uint8 --> float32
    res = input.astype('float32')

    # Normalize
    mean = (0.485, 0.456, 0.406)
    sd = (0.229, 0.224, 0.225)
    for i in range(3):
        res[i] = (res[i] / 255  - mean[i]) / sd[i]
    
    return res


# input: list<str>
# output: tensor
# 输入包括文件名的数组，输出一个 ndarry 对象
def gen_batch(filename_list, is_raw=False):
   
    # 图片读取
    if is_raw:
        img_list = filename_list
    else:
        img_list = [imread(name) for name in filename_list]

    # rezize and crop
    # img_list = [img_preprocess(img) for img in img_list]

    # 改变数组形状 (32, 32, 3) --> (3, 32, 32)
    img_list = [np.transpose(img, (2, 0, 1)) for img in img_list]

    # uint8 --> float32
    # Normalize ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_list = [array_normalize(img) for img in img_list]

    # 拼接 image --> [N, C, H, W]
    output = np.stack(img_list, 0)

    return output


# 推理自定义一个 Batch 图片
# input: ndarray
# output: list<str>
# 输入一个张量，返回推理结果的 list
@program_run_time
def cpu_ort_infer_batch(input, key, return_dict):

    # 加载模型
    session = onnxruntime.InferenceSession(LOCATION)
    # session = onnxruntime.InferenceSession("resnet50.onnx")

    # ONNX Runtime 推理过程
    ort_inputs = {session.get_inputs()[0].name: input}
    ort_outs = session.run(None, ort_inputs)
    out_prob = ort_outs[0]

    label_id = np.argmax(out_prob, 1)
    predicted_index = label_id.tolist()
    res = [imagenet_labels[i] for i in predicted_index]
    # res = predicted_index
    return_dict[key] = res
    return res


# input: directory name
# outut: inference resule
# 从文件夹读取图片，确定 batchsize, 再进行推理
@program_run_time
def cpu_ort_inference(dirname='./mini_batch'):
    # 文件名列表
    img_list = os.listdir(dirname)
    img_list = [os.path.join(dirname, name) for name in img_list]

    # 文件夹中文件总数
    count = len(img_list)

    # batch size
    batch_size = 10

    # 将文件名数组转化为多维数组
    res = [img_list[i: i + batch_size] for i in range(0, count, batch_size)]

    predicted = []

    # 开始循环推理
    for images in res:
        input = gen_batch(images)
        tmp = cpu_ort_infer_batch(input)
        log('predicted is: ', tmp)
        predicted.append(tmp)

    return predicted


if __name__ == "__main__":
    
    cpu_ort_inference('./mini_batch') 
    # cpu_ort_inference()
