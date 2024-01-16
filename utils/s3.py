
import boto3
import os
import uuid

from utils.constant import aws_access_key_id, aws_secret_access_key

S3_BUCKET = 'ai-space'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
url_profix = "https://cdn.aispace.ink/%s"
INFERENCE_SUFFIX = "completed/%s/image/%s.png"


async def upload(file_path, uid):
    async def upload(file_path, key):
        try:
            with open(file_path, 'rb') as data:
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=data,
                    ACL='public-read',
                )
        except Exception as e:
            raise e
        return url_profix % key

    key = INFERENCE_SUFFIX % (uid, uuid.uuid4().hex[:8])
    await upload(file_path, key)
    return url_profix % key
