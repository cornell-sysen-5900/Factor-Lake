"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/deploy_backtest_lambda.py
PURPOSE: Creates or updates the backtest Lambda, its IAM role and the 90-day result expiry rule.
VERSION: 1.1.0

Usage (from the project root, with AWS keys in .env or the environment):
    uv run python scripts/deploy_backtest_lambda.py              # deploy, then ping the function
    uv run python scripts/deploy_backtest_lambda.py --role-arn <arn>   # keys without IAM access
    uv run python scripts/deploy_backtest_lambda.py --package-only build/backtest_lambda.zip
    # A separate dev function that leaves the bucket's lifecycle rules alone (only checks them):
    uv run python scripts/deploy_backtest_lambda.py --function-name sysen5900-factor-lake-backtest-dev \
        --role-arn <the course role's ARN> --leave-lifecycle-rules

It is safe to run again: it updates what exists and creates what is missing.

What it sets up:
- IAM role factor-lake-backtest-lambda (unless --role-arn names an existing one): read
  the data under the S3 prefix, write results under <prefix>/backtest-results/, and
  write CloudWatch logs.
- Lambda factor-lake-backtest (Python 3.12) running aws_lambda/backtest_handler.py.
  pandas, numpy and pyarrow are bundled in the package at the versions this
  environment runs, so Lambda computes with the same libraries as the app (and no
  Lambda layer permission is needed). The package is too big for a direct upload,
  so it goes to S3 first.
- An S3 lifecycle rule deleting saved results after 90 days. Other lifecycle rules
  on the bucket are kept.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - runs uv with fixed arguments to fetch the Lambda's libraries
import sys
import tempfile
import time
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_service import RESULT_RETENTION_DAYS, RESULTS_DIR  # noqa: E402
from src.lambda_client import DEFAULT_FUNCTION_NAME  # noqa: E402
from src.s3_client import DEFAULT_BUCKET, DEFAULT_PREFIX, DEFAULT_REGION  # noqa: E402
from src.wrds_data import WRDS_DIR_ENV, wrds_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROLE_NAME = 'factor-lake-backtest-lambda'
RUNTIME = 'python3.12'
HANDLER = 'aws_lambda.backtest_handler.handler'
LIFECYCLE_RULE_ID = 'factor-lake-expire-backtest-results'
BASIC_EXECUTION_POLICY = 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
# Source the Lambda needs; supabase_client is left out (the supabase package is not in the Lambda)
PACKAGE_DIRS = ('src', 'aws_lambda')
EXCLUDED_FILES = {'src/supabase_client.py'}
# Libraries bundled into the package (boto3 is part of the Lambda runtime)
BUNDLED_LIBRARIES = ('pandas', 'numpy', 'pyarrow')
# Lambda's Python 3.12 runs on Amazon Linux 2023 (glibc 2.34)
LAMBDA_PLATFORM = 'x86_64-manylinux_2_28'
# Parts of the libraries the backtest never loads, removed to stay under Lambda's 250 MB limit.
# pyarrow's Flight RPC libraries are not needed by any Parquet module (its Substrait library is,
# so it stays); the rest are test suites, C/C++ headers and Cython sources.
PRUNED_PATHS = ('pyarrow/libarrow_flight.so*', 'pyarrow/libarrow_python_flight.so*', 'pyarrow/_flight.*',
                'pyarrow/include', 'pyarrow/tests', 'pyarrow/_pyarrow_cpp_tests.*',
                'pandas/tests', 'numpy/tests', 'numpy/*/tests')
PRUNED_SUFFIXES = ('.pyx', '.pxd', '.pxi', '.h', '.hpp', '.cc', '.cpp')
LAMBDA_UNZIPPED_LIMIT = 250 * 1024 * 1024
# Where the package is uploaded before Lambda is pointed at it
PACKAGE_DIR = '_deploy'

TRUST_POLICY = {
    'Version': '2012-10-17',
    'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'lambda.amazonaws.com'}, 'Action': 'sts:AssumeRole'}],
}


def install_libraries(target: Path) -> None:
    """Installs Linux builds of the bundled libraries, at this environment's versions, into target."""
    pins = [f"{name}=={version(name)}" for name in BUNDLED_LIBRARIES]
    logger.info(f"Fetching {', '.join(pins)} for {LAMBDA_PLATFORM} / {RUNTIME}")
    subprocess.run(  # nosec B603 - fixed command, versions come from installed package metadata
        [shutil.which('uv') or 'uv', 'pip', 'install', '--quiet', '--target', str(target),
         '--python-platform', LAMBDA_PLATFORM, '--python-version', RUNTIME.removeprefix('python'),
         '--only-binary', ':all:', *pins],
        check=True,
    )
    prune_libraries(target)


def prune_libraries(target: Path) -> None:
    """Deletes the library parts listed in PRUNED_PATHS / PRUNED_SUFFIXES."""
    for pattern in PRUNED_PATHS:
        for path in target.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for path in list(target.rglob('*')):
        if path.is_file() and (path.suffix in PRUNED_SUFFIXES or '__pycache__' in path.parts):
            path.unlink()


def build_package(libraries: Optional[Path] = None) -> bytes:
    """Zips the Python files the Lambda runs (src/ and aws_lambda/), plus the libraries in `libraries`."""
    buffer = io.BytesIO()
    unzipped = 0
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in PACKAGE_DIRS:
            for path in sorted((PROJECT_ROOT / folder).glob('*.py')):
                name = path.relative_to(PROJECT_ROOT).as_posix()
                if name not in EXCLUDED_FILES:
                    zf.write(path, name)
                    unzipped += path.stat().st_size
        if libraries is not None:
            for path in sorted(libraries.rglob('*')):
                if path.is_file():
                    zf.write(path, path.relative_to(libraries).as_posix())
                    unzipped += path.stat().st_size
    if unzipped > LAMBDA_UNZIPPED_LIMIT:
        raise SystemExit(f"The package is {unzipped / 1e6:.0f} MB unzipped; Lambda allows 250 MB. "
                         f"Add more unused library parts to PRUNED_PATHS.")
    logger.info(f"Package: {len(buffer.getvalue()) / 1e6:.1f} MB zipped, {unzipped / 1e6:.0f} MB unzipped")
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


def role_setup_instructions(role_name: str, bucket: str, prefix: str) -> str:
    """What an AWS admin needs to create when the deploying keys have no IAM access."""
    return (
        f"Please create an IAM role named {role_name} for AWS Lambda:\n\n"
        f"1. Trust policy (who may use the role):\n{json.dumps(TRUST_POLICY, indent=2)}\n\n"
        f"2. Attach the AWS managed policy {BASIC_EXECUTION_POLICY} (CloudWatch logs).\n\n"
        f"3. Add this inline policy (S3 access):\n{json.dumps(role_policy(bucket, prefix), indent=2)}\n\n"
        f"4. Allow the deploying user iam:PassRole on the role, plus lambda:CreateFunction, "
        f"lambda:UpdateFunctionCode, lambda:UpdateFunctionConfiguration, lambda:GetFunction, "
        f"lambda:InvokeFunction and s3:GetLifecycleConfiguration / s3:PutLifecycleConfiguration on {bucket}.\n\n"
        f"Then deploy with: uv run python scripts/deploy_backtest_lambda.py --role-arn <the role's ARN>"
    )


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


def upload_package(s3, bucket: str, prefix: str, package: bytes) -> str:
    """Uploads the package to S3 (named by its hash) and returns its key."""
    key = f"{prefix}/{PACKAGE_DIR}/backtest-lambda-{hashlib.sha256(package).hexdigest()[:16]}.zip"
    s3.upload_fileobj(io.BytesIO(package), bucket, key)
    logger.info(f"Uploaded the package to s3://{bucket}/{key}")
    return key


def deploy_function(lam, name: str, role_arn: str, code: Dict[str, str],
                    memory: int, timeout: int, environment: Dict[str, str]) -> None:
    """Creates the function, or updates its code and settings when it exists. code: S3Bucket and S3Key."""
    from botocore.exceptions import ClientError
    config = dict(Role=role_arn, Handler=HANDLER, Runtime=RUNTIME, Layers=[],
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
        lam.update_function_code(FunctionName=name, **code)
        lam.get_waiter('function_updated_v2').wait(FunctionName=name)
        lam.update_function_configuration(FunctionName=name, **config)
        lam.get_waiter('function_updated_v2').wait(FunctionName=name)
        logger.info(f"Updated Lambda {name}")
        return

    # A new role takes a few seconds before Lambda can use it
    for attempt in range(10):
        try:
            lam.create_function(FunctionName=name, Code=code, **config)
            break
        except ClientError as e:
            if 'cannot be assumed' not in str(e) or attempt == 9:
                raise
            time.sleep(5)
    lam.get_waiter('function_active_v2').wait(FunctionName=name)
    logger.info(f"Created Lambda {name}")


def set_result_expiry(s3, bucket: str, prefix: str, days: int) -> bool:
    """
    Adds (or updates) the rule deleting saved results after `days` days, keeping other
    rules. Returns False (with instructions) when these keys may not change lifecycle rules.
    """
    from botocore.exceptions import ClientError
    try:
        try:
            existing = s3.get_bucket_lifecycle_configuration(Bucket=bucket)['Rules']
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchLifecycleConfiguration':
                raise
            existing = []
        rules = merge_lifecycle_rules(existing, prefix, days)
        s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={'Rules': rules})
    except ClientError as e:
        if e.response['Error']['Code'] not in ('AccessDenied', 'AccessDeniedException'):
            raise
        logger.warning(f"These keys may not change the bucket's lifecycle rules, so saved results will not "
                       f"expire yet. Ask an AWS admin to add an S3 lifecycle rule on s3://{bucket}: prefix "
                       f"'{prefix}/{RESULTS_DIR}/', expire current versions after {days} days.")
        return False
    logger.info(f"s3://{bucket}/{prefix}/{RESULTS_DIR}/ results now expire after {days} days "
                f"({len(rules) - 1} other lifecycle rule(s) kept)")
    return True


def expiry_rule_covers(rules: List[Dict[str, Any]], prefix: str) -> bool:
    """
    True when an enabled lifecycle rule expires everything under <prefix>/backtest-results/:
    the result JSON and the WRDS holdings Parquet files both live there.
    """
    results = f'{prefix}/{RESULTS_DIR}/'
    for rule in rules:
        rule_prefix = rule.get('Filter', {}).get('Prefix', rule.get('Prefix'))
        if (rule.get('Status') == 'Enabled' and rule_prefix is not None and results.startswith(rule_prefix)
                and rule.get('Expiration', {}).get('Days')):
            return True
    return False


def check_result_expiry(s3, bucket: str, prefix: str) -> bool:
    """Reads (never writes) the bucket's lifecycle rules and says whether saved results expire."""
    from botocore.exceptions import ClientError
    try:
        rules = s3.get_bucket_lifecycle_configuration(Bucket=bucket)['Rules']
    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'NoSuchLifecycleConfiguration':
            rules = []
        elif code in ('AccessDenied', 'AccessDeniedException'):
            logger.warning(f"These keys may not read the lifecycle rules of s3://{bucket}; ask an AWS admin to "
                           f"check that '{prefix}/{RESULTS_DIR}/' expires.")
            return False
        else:
            raise
    if expiry_rule_covers(rules, prefix):
        logger.info(f"Lifecycle rules left as they are; s3://{bucket}/{prefix}/{RESULTS_DIR}/ (results and "
                    f"holdings) expires.")
        return True
    logger.warning(f"No enabled lifecycle rule expires s3://{bucket}/{prefix}/{RESULTS_DIR}/.")
    return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy the Factor-Lake backtest Lambda.")
    parser.add_argument('--function-name', default=DEFAULT_FUNCTION_NAME)
    parser.add_argument('--role-name', default=ROLE_NAME)
    parser.add_argument('--region', default=os.environ.get('AWS_DEFAULT_REGION') or DEFAULT_REGION)
    parser.add_argument('--bucket', default=os.environ.get('FACTOR_LAKE_S3_BUCKET') or DEFAULT_BUCKET)
    parser.add_argument('--prefix', default=os.environ.get('FACTOR_LAKE_S3_PREFIX', DEFAULT_PREFIX))
    parser.add_argument('--memory', type=int, default=3008, help="MB (default 3008)")
    parser.add_argument('--timeout', type=int, default=600, help="Seconds (default 600)")
    parser.add_argument('--package-only', type=Path, metavar='ZIP', help="Only write the code package here")
    parser.add_argument('--role-arn', help="Use this existing role and make no IAM calls (for keys without IAM access)")
    parser.add_argument('--print-role-policy', action='store_true',
                        help="Print the role an AWS admin needs to create, and exit")
    parser.add_argument('--wrds-dir', default=wrds_dir(),
                        help=f"Folder of the WRDS data under the prefix (default {wrds_dir()})")
    parser.add_argument('--leave-lifecycle-rules', action='store_true',
                        help="Do not write the bucket's lifecycle rules (e.g. for a dev function); only check "
                             "that the result expiry rule exists")
    args = parser.parse_args(argv)
    prefix = args.prefix.strip('/')

    if args.print_role_policy:
        print(role_setup_instructions(args.role_name, args.bucket, prefix))
        return 0

    with tempfile.TemporaryDirectory(prefix='factor_lake_lambda_') as tmp:
        install_libraries(Path(tmp))
        package = build_package(Path(tmp))
    if args.package_only:
        args.package_only.parent.mkdir(parents=True, exist_ok=True)
        args.package_only.write_bytes(package)
        logger.info(f"Wrote {args.package_only}; nothing deployed.")
        return 0

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    import boto3

    lam = boto3.client('lambda', region_name=args.region)
    s3 = boto3.client('s3', region_name=args.region)

    if args.role_arn:
        role_arn = args.role_arn
        logger.info(f"Using IAM role {role_arn} (no IAM calls)")
    else:
        from botocore.exceptions import ClientError
        try:
            role_arn = ensure_role(boto3.client('iam'), args.role_name, args.bucket, prefix)
        except ClientError as e:
            if e.response['Error']['Code'] not in ('AccessDenied', 'AccessDeniedException'):
                raise
            logger.error("These AWS keys may not use IAM, so the script cannot create the Lambda's role.\n"
                         "Ask an AWS admin to create it (run with --print-role-policy for what to send them),\n"
                         "then deploy with --role-arn <the role's ARN>.")
            return 1

    key = upload_package(s3, args.bucket, prefix, package)
    deploy_function(lam, args.function_name, role_arn, {'S3Bucket': args.bucket, 'S3Key': key},
                    args.memory, args.timeout,
                    {'FACTOR_LAKE_S3_BUCKET': args.bucket, 'FACTOR_LAKE_S3_PREFIX': prefix,
                     WRDS_DIR_ENV: args.wrds_dir.strip('/')})
    if args.leave_lifecycle_rules:
        expiry_set = check_result_expiry(s3, args.bucket, prefix)
    else:
        expiry_set = set_result_expiry(s3, args.bucket, prefix, RESULT_RETENTION_DAYS)

    response = lam.invoke(FunctionName=args.function_name, Payload=json.dumps({'action': 'ping'}).encode())
    reply = json.loads(response['Payload'].read())
    if response.get('FunctionError') or reply.get('status') != 'ok':
        logger.error(f"Ping failed: {reply}")
        return 1
    logger.info(f"Ping OK (backtest code version {reply['code_version']}). "
                f"Set FACTOR_LAKE_BACKTEST_LAMBDA={args.function_name} for the app."
                + ("" if expiry_set else " (Result expiry still needs an admin, see the warning above.)"))
    return 0


if __name__ == '__main__':
    sys.exit(main())
