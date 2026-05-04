#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/archive_supabase_tables.py
PURPOSE: Archiving Supabase tables to Parquet format and pushing to GitHub.
"""

import os
import sys
import logging
import urllib.request
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseArchiver:
    def __init__(self):
        self.url = os.environ.get('SUPABASE_URL')
        self.key = os.environ.get('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
        
        self.client: Client = create_client(self.url, self.key)
        
        # Default to data/archive relative to the project root if not specified
        default_archive_path = Path(__file__).resolve().parent.parent / "data" / "archive"
        self.archive_dir = Path(os.environ.get('ARCHIVE_REPO_PATH', default_archive_path))
        
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def fetch_table_names(self) -> List[str]:
        """
        Dynamically fetches all table names exposed by the Supabase PostgREST API.
        """
        manual_tables = os.environ.get("TABLES_TO_ARCHIVE")
        if manual_tables:
            tables = [t.strip() for t in manual_tables.split(",") if t.strip()]
            logger.info(f"Using manually provided tables: {tables}")
            return tables

        logger.info("Fetching OpenAPI schema to discover tables...")
        req_url = f"{self.url.rstrip('/')}/rest/v1/"
        req = urllib.request.Request(req_url, headers={
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}"
        })
        
        try:
            with urllib.request.urlopen(req) as response:
                spec = json.loads(response.read())
                # PostgREST OpenAPI v2 exposes tables under 'definitions'
                tables = list(spec.get("definitions", {}).keys())
                logger.info(f"Discovered {len(tables)} tables/views: {tables}")
                return tables
        except urllib.error.HTTPError as e:
            if e.code == 401:
                logger.error(
                    "HTTP 401 Unauthorized when accessing OpenAPI schema. "
                    "This usually happens if you are using an anonymous key, "
                    "or if OpenAPI exposition is disabled in your Supabase project. "
                    "Please provide the SERVICE_ROLE_KEY as SUPABASE_KEY, or manually define the "
                    "tables to archive using the TABLES_TO_ARCHIVE environment variable "
                    "(e.g. TABLES_TO_ARCHIVE='users,orders' python3 script.py)."
                )
            else:
                logger.error(f"Failed to fetch table schema: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch table schema: {e}")
            raise

    def download_table_to_parquet(self, table_name: str) -> Path:
        """
        Downloads all rows from a given table using pagination and saves it as a Parquet file.
        """
        page_size = 1000
        offset = 0
        all_rows = []

        logger.info(f"Downloading table: {table_name}")
        try:
            while True:
                response = self.client.table(table_name).select('*').range(offset, offset + page_size - 1).execute()
                batch = response.data if hasattr(response, 'data') else []

                if not batch:
                    break
                
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                
                offset += page_size
        except Exception as e:
            logger.error(f"Failed to fetch data for {table_name}: {e}")
            raise

        df = pd.DataFrame(all_rows)
        
        # Ensure all columns are strings if they are objects, to prevent parquet type issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        output_path = self.archive_dir / f"{table_name}.parquet"
        
        if df.empty:
            logger.warning(f"Table {table_name} is empty. Creating an empty Parquet file.")
        
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path

    def create_github_release(self):
        """
        Creates a GitHub release using the 'gh' CLI and uploads all Parquet files as assets.
        This prevents bloating the Git repository size with large binary files.
        """
        logger.info(f"Initiating GitHub release creation in {self.archive_dir}")
        cwd = str(self.archive_dir)

        # Ensure gh CLI is available
        try:
            subprocess.run(['gh', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("GitHub CLI ('gh') is not installed or not found in PATH. Cannot create release.")
            raise RuntimeError("gh CLI required for releases")

        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tag_name = f"backup-{date_str}"
        title = f"Database Backup {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        notes = "Automated backup of Supabase tables."

        logger.info(f"Creating release {tag_name}...")
        
        # We need the repo context. If we're inside a git repo, gh will figure it out.
        # Alternatively, the user can set ARCHIVE_GITHUB_REPO="owner/repo" in the environment.
        repo = os.environ.get('ARCHIVE_GITHUB_REPO')
        
        cmd = ['gh', 'release', 'create', tag_name, '--title', title, '--notes', notes]
        if repo:
            cmd.extend(['--repo', repo])
            
        # Add all parquet files as assets
        parquet_files = list(self.archive_dir.glob('*.parquet'))
        if not parquet_files:
            logger.warning("No parquet files found to release.")
            return
            
        for p in parquet_files:
            cmd.append(str(p.name))
            
        try:
            # gh release create requires the tag to exist locally or on remote, or it will create it on the default branch.
            res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Failed to create GitHub release: {res.stderr}")
                raise RuntimeError(f"GitHub release failed: {res.stderr}")
            else:
                logger.info(f"Successfully created GitHub release {tag_name} with {len(parquet_files)} assets.")
        except subprocess.CalledProcessError as e:
            logger.error(f"GitHub release command failed: {e}")
            raise

    def run(self):
        """
        Main execution flow.
        """
        tables = self.fetch_table_names()
        for table in tables:
            self.download_table_to_parquet(table)
        self.create_github_release()


if __name__ == "__main__":
    try:
        archiver = SupabaseArchiver()
        archiver.run()
    except Exception as e:
        logger.error(f"Archiver failed: {e}")
        sys.exit(1)
