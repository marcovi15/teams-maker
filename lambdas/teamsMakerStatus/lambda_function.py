import boto3
import json

def lambda_handler(event, context):
    request_id = event['queryStringParameters']['request_id']
    s3 = boto3.client('s3')

    try:
        response = s3.get_object(Bucket='footyapp-status-container-while-script-runs', Key=f'status/{request_id}.json')
        extracted_res = response['Body'].read().decode('utf-8')
        data = json.loads(extracted_res)

        result = {
            'statusCode': data.get('statusCode', 200),
            'headers': {"Content-Type": "application/json"},
            'body': json.dumps({
                'status': data.get('status', 'done'),
                'result': data.get('body', data['body'])
            })
        }
    except s3.exceptions.NoSuchKey:
        result = {
            'statusCode': 202,
            'headers': {"Content-Type": "application/json"},
            'body': json.dumps({
                'status': 'pending',
                'result': ''
            })
        }

    return result
