# command for create cloudformation stack
aws cloudformation create-stack --stack-name shen-test --template-url https://dl-model-bucket-shen.s3.ap-southeast-1.amazonaws.com/DeepLearning_Serverless_CF.json --parameters ParameterKey=ModelAndCodeBucket,ParameterValue=dl-model-bucket-shen ParameterKey=DLBundleObjectKey,ParameterValue=resnet50.zip ParameterKey=myFunctionName,ParameterValue=Ort-resnet50 --capabilities CAPABILITY_NAMED_IAM
# command for get function url
aws lambda get-function-url-config --function-name Ort-resnet50


# command for get log groups
aws logs describe-log-groups 
# command for get log streams
aws logs describe-log-streams --log-group-name /aws/lambda/Ort-resnet50
# command for get stream logs
aws logs get-log-events --log-group-name "/aws/lambda/Ort-resnet50" --log-stream-name "2022/10/25/[\$LATEST]6d726e6439de4dcead57435a1e7bf31e"

# command for delete log groups
aws logs delete-log-group --log-group-name /aws/lambda/Ort-resnet50