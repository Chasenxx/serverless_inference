# serverless_inference

## 1. batch request 
```json
{
    "concurrency": 2,
    "batchsize": 5,
    "size": 10,
    "file-type": "image", 
    "data": ["data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8", "data9", "data10"],
    "isBase64Encoded": true
}
```

## 2. batch response 
```json
{
    "concurrency": 2,
    "batchsize": 5,
    "size": 10,
    "inference": []
}
```