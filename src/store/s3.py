import boto3
import os
from pathlib import Path

def upload_folder_to_s3(local_folder: str, bucket_name: str, s3_prefix: str, aws_profile: str = None):
    """
    Upload all files in a local folder (recursively) to an S3 bucket under a given prefix.
    Args:
        local_folder (str): Path to the local folder to upload
        bucket_name (str): Name of the S3 bucket
        s3_prefix (str): S3 key prefix (folder path in bucket)
        aws_profile (str, optional): AWS profile to use (if not default)
    """
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client('s3')
    local_folder = Path(local_folder)
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = Path(root) / file
            rel_path = local_path.relative_to(local_folder)
            s3_key = f"{s3_prefix}/{rel_path.as_posix()}"
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(str(local_path), bucket_name, s3_key)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload parsed data to S3 bucket")
    parser.add_argument("--local_folder", type=str, required=True, help="Local folder to upload (e.g. data/parsed/NVDA/Q4FY25-CFO-Commentary)")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--prefix", type=str, required=True, help="S3 prefix/folder (e.g. assignment_03/parsed/FNTBX)")
    parser.add_argument("--profile", type=str, default=None, help="AWS profile name (optional)")
    args = parser.parse_args()
    upload_folder_to_s3(args.local_folder, args.bucket, args.prefix, args.profile)
