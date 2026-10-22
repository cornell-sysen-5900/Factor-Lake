"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/deploy_backtest_lambda.py
PURPOSE: Creates or updates the backtest Lambda, its IAM role and the 90-day result expiry rule.
VERSION: 1.0.0

Usage (from the project root, with AWS keys in .env or the environment):
    uv run python scripts/deploy_backtest_lambda.py              # deploy, then ping the function
    uv run python scripts/deploy_backtest_lambda.py --package-only build/backtest_lambda.zip

It is safe to run again: it updates what exists and creates what is missing.

What it sets up:
- IAM role factor-lake-backtest-lambda: read the data under the S3 prefix, write
  results under <prefix>/backtest-results/, and write CloudWatch logs.
- Lambda factor-lake-backtest (Python 3.12) running aws_lambda/backtest_handler.py,
  with pandas, numpy and pyarrow from the AWS SDK for pandas layer, so the upload is
  only this project's code.
- An S3 lifecycle rule deleting saved results after 90 days. Other lifecycle rules
  on the bucket are kept.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_service import RESULT_RETENTION_DAYS, RESULTS_DIR  # noqa: E402
from src.lambda_client import DEFAULT_FUNCTION_NAME  # noqa: E402
from src.s3_client import DEFAULT_BUCKET, DEFAULT_PREFIX, DEFAULT_REGION  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROLE_NAME = 'factor-lake-backtest-lambda'
RUNTIME = 'python3.12'
HANDLER = 'aws_lambda.backtest_handler.handler'
# AWS SDK for pandas: AWS's public layer with pandas, numpy and pyarrow (https://aws-sdk-pandas.readthedocs.io)
PANDAS_LAYER = 'arn:aws:lambda:{region}:336392948345:layer:AWSSDKPandas-Python312'
LIFECYCLE_RULE_ID = 'factor-lake-expire-backtest-results'
BASIC_EXECUTION_POLICY = 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
# Source the Lambda needs; supabase_client is left out (the supabase package is not in the Lambda)
PACKAGE_DIRS = ('src', 'aws_lambda')
EXCLUDED_FILES = {'src/supabase_client.py'}

TRUST_POLICY = {
    'Version': '2012-10-17',
    'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'lambda.amazonaws.com'}, 'Action': 'sts:AssumeRole'}],
}


def build_package() -> bytes:
    """Zips the Python files the Lambda runs (src/ and aws_lambda/)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in PACKAGE_DIRS:
            for path in sorted((PROJECT_ROOT / folder).glob('*.py')):
                name = path.relative_to(PROJECT_ROOT).as_posix()
                if name not in EXCLUDED_FILES:
                    zf.write(path, name)
    return buffer.getvalue()


def role_policy(bucket: str, prefix: str) -> Dict[str, Any]:
    """
    What the Lambda may do in S3: read everything under the prefix and write only saved
    results. ListBucket (limited to the prefix) makes S3 answer "not found" for a result
    that is not saved yet, instead of "access denied".
    """
    return {
        'Version': '2012-10-17',
        'Statement': [
            {'Effect': 'Allow', 'Action': ['s3:GetObject'],
             'Resource': f'arn:aws:s3:::{bucket}/{prefix}/*'},
            {'Effect': 'Allow', 'Action': ['s3:PutObject'],
             'Resource': f'arn:aws:s3:::{bucket}/{prefix}/{RESULTS_DIR}/*'},
            {'Effect': 'Allow', 'Action': ['s3:ListBucket'],
             'Resource': f'arn:aws:s3:::{bucket}',
             'Condition': {'StringLike': {'s3:prefix': [f'{prefix}/*']}}},
        ],
    }


def merge_lifecycle_rules(existing: List[Dict[str, Any]], prefix: str, days: int) -> List[Dict[str, Any]]:
    """
    The bucket's lifecycle rules with ours added or replaced. S3 replaces the whole
    lifecycle configuration on every write, so the other rules must be sent back too.
    """
    ours = {
        'ID': LIFECYCLE_RULE_ID,
        'Filter': {'Prefix': f'{prefix}/{RESULTS_DIR}/'},
        'Status': 'Enabled',
        'Expiration': {'Days': int(days)},
    }
    return [rule for rule in existing if rule.get('ID') != LIFECYCLE_RULE_ID] + [ours]


def ensure_role(iam, role_name: str, bucket: str, prefix: str) -> str:
    """Creates the Lambda's role if needed, sets its permissions, and returns its ARN."""
    from botocore.exceptions import ClientError
    try:
        arn = iam.get_role(RoleName=role_name)['Role']['Arn']
        logger.info(f"Using IAM role {role_name}")
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchEntity':
            raise
        arn = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
                              Description='Factor-Lake backtest Lambda')['Role']['Arn']
        logger.info(f"Created IAM role {role_name}")
    iam.attach_role_policy(RoleName=role_name, PolicyArn=BASIC_EXECUTION_POLICY)
    iam.put_role_policy(RoleName=role_name, PolicyName='factor-lake-s3-access',
                        PolicyDocument=json.dumps(role_policy(bucket, prefix)))
    return arn


def resolve_layer(lam, region: str, layer_arn: Optional[str]) -> str:
    """The pandas layer version to use: --layer-arn, else the newest published version."""
    if layer_arn:
        return layer_arn
    base = PANDAS_LAYER.format(region=region)
    try:
        versions = lam.list_layer_versions(LayerName=base, CompatibleRuntime=RUNTIME)['LayerVersions']
    except Exception as e:
        raise SystemExit(f"Could not look up {base} ({e}). Pass --layer-arn with a version from "
                         f"https://aws-sdk-pandas.readthedocs.io/en/stable/layers.html") from e
    if not versions:
        raise SystemExit(f"No versions of {base} found; pass --layer-arn.")
    return max(versions, key=lambda v: v['Version'])['LayerVersionArn']


def deploy_function(lam, name: str, role_arn: str, package: bytes, layer_arn: str,
                    memory: int, timeout: int, environment: Dict[str, str]) -> None:
    """Creates the function, or updates its code and settings when it exists."""
    from botocore.exceptions import ClientError
    config = dict(Role=role_arn, Handler=HANDLER, Runtime=RUNTIME, Layers=[layer_arn],
                  MemorySize=memory, Timeout=timeout, Environment={'Variables': environment},
                  Description='Factor-Lake backtests: reads market data from S3, caches results in S3')
    try:
        lam.get_function(FunctionName=name)
        exists = True
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceNotFoundException':
            raise
        exists = False

    if exists:
        lam.update_function_code(FunctionName=name, ZipFile=package)
        lam.get_waiter('function_updated_v2').wait(FunctionName=name)
        lam.update_function_configuration(FunctionName=name, **config)
        lam.get_waiter('function_updated_v2').wait(FunctionName=name)
        logger.info(f"Updated Lambda {name}")
        return

    # A new role takes a few seconds before Lambda can use it
    for attempt in range(10):
        try:
            lam.create_function(FunctionName=name, Code={'ZipFile': package}, **config)
            break
        except ClientError as e:
            if 'cannot be assumed' not in str(e) or attempt == 9:
                raise
            time.sleep(5)
    lam.get_waiter('function_active_v2').wait(FunctionName=name)
    logger.info(f"Created Lambda {name}")


def set_result_expiry(s3, bucket: str, prefix: str, days: int) -> None:
    """Adds (or updates) the rule deleting saved results after `days` days, keeping other rules."""
    from botocore.exceptions import ClientError
    try:
        existing = s3.get_bucket_lifecycle_configuration(Bucket=bucket)['Rules']
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchLifecycleConfiguration':
            raise
        existing = []
    rules = merge_lifecycle_rules(existing, prefix, days)
    s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={'Rules': rules})
    logger.info(f"s3://{bucket}/{prefix}/{RESULTS_DIR}/ results now expire after {days} days "
                f"({len(rules) - 1} other lifecycle rule(s) kept)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy the Factor-Lake backtest Lambda.")
    parser.add_argument('--function-name', default=DEFAULT_FUNCTION_NAME)
    parser.add_argument('--role-name', default=ROLE_NAME)
    parser.add_argument('--region', default=os.environ.get('AWS_DEFAULT_REGION') or DEFAULT_REGION)
    parser.add_argument('--bucket', default=os.environ.get('FACTOR_LAKE_S3_BUCKET') or DEFAULT_BUCKET)
    parser.add_argument('--prefix', default=os.environ.get('FACTOR_LAKE_S3_PREFIX', DEFAULT_PREFIX))
    parser.add_argument('--memory', type=int, default=3008, help="MB (default 3008)")
    parser.add_argument('--timeout', type=int, default=600, help="Seconds (default 600)")
    parser.add_argument('--layer-arn', help="pandas layer version ARN (default: newest AWSSDKPandas-Python312)")
    parser.add_argument('--package-only', type=Path, metavar='ZIP', help="Only write the code package here")
    args = parser.parse_args(argv)
    prefix = args.prefix.strip('/')

    package = build_package()
    logger.info(f"Code package: {len(package) / 1024:.0f} KB")
    if args.package_only:
        args.package_only.parent.mkdir(parents=True, exist_ok=True)
        args.package_only.write_bytes(package)
        logger.info(f"Wrote {args.package_only}; nothing deployed.")
        return 0

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    import boto3

    iam = boto3.client('iam')
    lam = boto3.client('lambda', region_name=args.region)
    s3 = boto3.client('s3', region_name=args.region)

    role_arn = ensure_role(iam, args.role_name, args.bucket, prefix)
    layer_arn = resolve_layer(lam, args.region, args.layer_arn)
    logger.info(f"pandas layer: {layer_arn}")
    deploy_function(lam, args.function_name, role_arn, package, layer_arn, args.memory, args.timeout,
                    {'FACTOR_LAKE_S3_BUCKET': args.bucket, 'FACTOR_LAKE_S3_PREFIX': prefix})
    set_result_expiry(s3, args.bucket, prefix, RESULT_RETENTION_DAYS)

    response = lam.invoke(FunctionName=args.function_name, Payload=json.dumps({'action': 'ping'}).encode())
    reply = json.loads(response['Payload'].read())
    if response.get('FunctionError') or reply.get('status') != 'ok':
        logger.error(f"Ping failed: {reply}")
        return 1
    logger.info(f"Ping OK (backtest code version {reply['code_version']}). "
                f"Set FACTOR_LAKE_BACKTEST_LAMBDA={args.function_name} for the app.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
