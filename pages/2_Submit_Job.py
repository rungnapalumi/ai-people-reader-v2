import os
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3


st.title("Video Analysis")

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    st.error("AWS_BUCKET not set")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)


def new_id():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:5]}"


uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "m4v"])

if st.button("Run Analysis"):
    if not uploaded:
        st.warning("upload first")
        st.stop()

    group_id = new_id()

    input_key = f"jobs/groups/{group_id}/input/input.mp4"

    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=input_key,
        Body=uploaded.getvalue(),
        ContentType="video/mp4",
    )

    outputs = {
        "dots": f"jobs/output/groups/{group_id}/dots.mp4",
        "skeleton": f"jobs/output/groups/{group_id}/skeleton.mp4",
        "report": f"jobs/output/groups/{group_id}/report.docx",
    }

    jobs = [
        {"mode": "dots"},
        {"mode": "skeleton"},
        {"mode": "report"},
    ]

    for j in jobs:
        job = {
            "job_id": new_id(),
            "group_id": group_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "mode": j["mode"],
            "input_key": input_key,
            "output_key": outputs.get(j["mode"], outputs["report"]),
        }

        key = f"jobs/pending/{job['job_id']}.json"
        s3.put_object(
            Bucket=AWS_BUCKET,
            Key=key,
            Body=json.dumps(job).encode(),
            ContentType="application/json",
        )

    st.success("submitted")
