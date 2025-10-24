from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import subprocess
import logging
import json

def get_parsed_folders():
    """Dynamically discover folders in data/parsed/ directory."""
    logger = logging.getLogger(__name__)
    base_dir = '/opt/airflow/workspace'
    parsed_data_path = os.path.join(base_dir, 'data', 'parsed')
    
    valid_folders = []
    
    # Check if parsed data directory exists
    if not os.path.exists(parsed_data_path):
        logger.error(f"Parsed data directory not found: {parsed_data_path}")
        return []
    
    # Discover folders in data/parsed/
    try:
        for folder_name in os.listdir(parsed_data_path):
            folder_path = os.path.join(parsed_data_path, folder_name)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                logger.debug(f"Skipping non-directory item: {folder_name}")
                continue
            
            # Skip hidden directories
            if folder_name.startswith('.'):
                logger.debug(f"Skipping hidden directory: {folder_name}")
                continue
            
            # Check if there are any files to upload
            has_content = False
            try:
                for root, dirs, files in os.walk(folder_path):
                    if files:
                        has_content = True
                        break
            except Exception as e:
                logger.warning(f"Error checking content in {folder_name}: {e}")
                continue
            
            if not has_content:
                logger.warning(f"No content found in {folder_name}, skipping")
                continue
            
            valid_folders.append(folder_name)
            logger.info(f"Valid folder found for storing: {folder_name}")
    
    except Exception as e:
        logger.error(f"Error discovering parsed folders: {e}")
        return []
    
    logger.info(f"Total valid folders found for storing: {len(valid_folders)}")
    return valid_folders

def run_storing(folder_name):
    """Upload parsed data for a given folder to S3."""
    logger = logging.getLogger(__name__)
    base_dir = '/opt/airflow/workspace'
    script_path = os.path.join(base_dir, 'src', 'store', 's3.py')
    local_folder = os.path.join(base_dir, 'data', 'parsed', folder_name)
    bucket = os.environ.get('S3_BUCKET_NAME', 'damg-binaryinsights-assignment-03')
    prefix = f'assignment_03/parsed'
    
    # Log bucket being used for debugging
    logger.info(f"Using S3 bucket from env: {bucket}")
    
    # Pre-flight checks
    if not os.path.exists(local_folder):
        raise Exception(f"Missing prerequisite: Parsed data directory not found at {local_folder}")
    
    if not os.path.exists(script_path):
        raise Exception(f"Script not found: {script_path}")
    
    # Check if there's actually content to upload
    has_content = False
    file_count = 0
    try:
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                file_count += 1
                has_content = True
        
        if not has_content:
            raise Exception(f"No files found to upload in {local_folder}")
        
        logger.info(f"Found {file_count} files to upload in {local_folder}")
    except Exception as e:
        raise Exception(f"Error checking content in {local_folder}: {e}")
    
    try:
        os.chdir(base_dir)
        logger.info(f"Uploading parsed data for {folder_name} to S3")
        logger.info(f"Local folder: {local_folder}")
        logger.info(f"S3 destination: s3://{bucket}/{prefix}")
        logger.info(f"Script path: {script_path}")
        
        # Log AWS credentials status (without showing secrets)
        aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
        logger.info(f"AWS_ACCESS_KEY_ID configured: {'Yes' if aws_key else 'NO - NOT SET'}")
        logger.info(f"AWS_SECRET_ACCESS_KEY configured: {'Yes' if aws_secret else 'NO - NOT SET'}")
        logger.info(f"AWS_DEFAULT_REGION: {os.environ.get('AWS_DEFAULT_REGION', 'not set')}")
        
        # Test boto3 availability
        try:
            import boto3
            logger.info(f"boto3 available: Yes (version {boto3.__version__})")
            
            # Test boto3 connection
            test_session = boto3.Session()
            test_client = test_session.client('s3', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))
            test_client.head_bucket(Bucket=bucket)
            logger.info(f"✅ S3 bucket accessible: {bucket}")
        except Exception as e:
            logger.error(f"❌ boto3 test failed: {e}")
            raise Exception(f"S3 connection test failed: {e}")
        
        # Run the s3.py script
        result = subprocess.run([
            'python', script_path,
            '--local_folder', local_folder,
            '--bucket', bucket,
            '--prefix', prefix
        ], capture_output=True, text=True, timeout=900)
        
        logger.info(f"S3 script exit code: {result.returncode}")
        logger.info(f"S3 script stdout:\n{result.stdout}")
        
        if result.returncode != 0:
            logger.error(f"S3 upload error for {folder_name}: {result.stderr}")
            raise Exception(f"Storing failed for {folder_name}. stderr: {result.stderr}")
        
        return f"✅ Storing Success: {folder_name}"
        
    except subprocess.TimeoutExpired:
        logger.error(f"S3 upload timeout for {folder_name}")
        raise Exception(f"Storing timeout for {folder_name}")
    except Exception as e:
        logger.error(f"❌ Storing error for {folder_name}: {str(e)}")
        raise

default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 7),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'storing_dag',
    default_args=default_args,
    description='Store parsed results in S3 for each company - dynamically discovers valid tickers',
    schedule_interval=None,
    catchup=False,
)

# Dynamically discover folders from data/parsed/
PARSED_FOLDERS = get_parsed_folders()

storing_tasks = []
for folder_name in PARSED_FOLDERS:
    storing_task = PythonOperator(
        task_id=f'store_reports_{folder_name.lower()}',
        python_callable=run_storing,
        op_args=[folder_name],
        dag=dag,
    )
    storing_tasks.append(storing_task)

# Log the discovered folders for debugging
logging.getLogger(__name__).info(f"DAG will process {len(PARSED_FOLDERS)} folders: {PARSED_FOLDERS}")
