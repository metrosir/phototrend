
import boto3
import os
import uuid

from utils.constant import aws_access_key_id, aws_secret_access_key

S3_BUCKET = 'ai-space'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
url_profix = "https://cdn.aispace.ink/%s"
# INFERENCE_SUFFIX = "completed/%s/image/%s.png"
dress_suffix = "aispace/app/user/dress/%s/%s/%s.png"


def upload(file_path, uid, worker_id=None):
    def upload(file_path, key):
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

    if worker_id is None:
        key = dress_suffix % (uid, uuid.uuid4().hex[:8], uuid.uuid4().hex[:8])
    else:
        key = dress_suffix % (uid, worker_id, uuid.uuid4().hex[:8])
    return upload(file_path, key)
