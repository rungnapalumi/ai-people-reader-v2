import os
import json
import time
import tempfile

import boto3

AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING = "jobs/pending/"
FINISHED = "jobs/finished/"
FAILED = "jobs/failed/"


def list_jobs():
    r = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING)
    return [x["Key"] for x in r.get("Contents", [])]


def load_json(key):
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode())


def move(src, dst_prefix):
    name = src.split("/")[-1]
    s3.copy_object(Bucket=AWS_BUCKET, CopySource=f"{AWS_BUCKET}/{src}", Key=dst_prefix + name)
    s3.delete_object(Bucket=AWS_BUCKET, Key=src)


def fake_video(out_key):
    s3.put_object(Bucket=AWS_BUCKET, Key=out_key, Body=b"video")


def fake_report(out_key):
    s3.put_object(Bucket=AWS_BUCKET, Key=out_key, Body=b"docx")


def process(job):
    mode = job["mode"]

    if mode == "dots":
        fake_video(job["output_key"])

    elif mode == "skeleton":
        fake_video(job["output_key"])

    elif mode == "report":
        fake_report(job["output_key"])

    else:
        raise Exception("unknown mode")


print("worker started")

while True:
    jobs = list_jobs()

    for key in jobs:
        try:
            job = load_json(key)
            process(job)
            move(key, FINISHED)
            print("done", job["mode"])

        except Exception as e:
            print("fail", e)
            move(key, FAILED)

    time.sleep(5)
