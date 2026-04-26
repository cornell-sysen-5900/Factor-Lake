import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

# Add the scripts directory to the path so we can import the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.archive_supabase_tables import SupabaseArchiver

class TestSupabaseArchiver(unittest.TestCase):
    @patch.dict(os.environ, {
        "SUPABASE_URL": "https://fake.supabase.co", 
        "SUPABASE_KEY": "fake_key",
        "ARCHIVE_REPO_PATH": "/tmp/fake_archive"
    })
    @patch("scripts.archive_supabase_tables.create_client")
    def setUp(self, mock_create_client):
        self.mock_supabase = MagicMock()
        mock_create_client.return_value = self.mock_supabase
        self.archiver = SupabaseArchiver()

    @patch("urllib.request.urlopen")
    def test_fetch_table_names(self, mock_urlopen):
        # Mock OpenAPI response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"definitions": {"users": {}, "orders": {}}}'
        # urlopen is used as a context manager
        mock_urlopen.return_value.__enter__.return_value = mock_response

        tables = self.archiver.fetch_table_names()
        
        self.assertEqual(tables, ["users", "orders"])
        mock_urlopen.assert_called_once()
        args, kwargs = mock_urlopen.call_args
        self.assertEqual(args[0].full_url, "https://fake.supabase.co/rest/v1/")
        self.assertEqual(args[0].headers.get('Apikey'), "fake_key")

    @patch("pandas.DataFrame.to_parquet")
    def test_download_table_to_parquet(self, mock_to_parquet):
        # Setup mock for Supabase pagination
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_range1 = MagicMock()
        mock_range2 = MagicMock()
        
        # Two pages of data
        mock_response1 = MagicMock()
        mock_response1.data = [{"id": 1, "name": "Alice"}] * 1000  # Full page
        mock_range1.execute.return_value = mock_response1
        
        mock_response2 = MagicMock()
        mock_response2.data = [{"id": 1001, "name": "Bob"}] * 50  # Partial page, ends pagination
        mock_range2.execute.return_value = mock_response2
        
        # Configure the chain: table().select().range().execute()
        self.mock_supabase.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.range.side_effect = [mock_range1, mock_range2]

        output_path = self.archiver.download_table_to_parquet("test_table")
        
        # Verify
        self.assertEqual(self.mock_supabase.table.call_args[0][0], "test_table")
        self.assertEqual(mock_select.range.call_count, 2)
        mock_to_parquet.assert_called_once()
        self.assertEqual(output_path, Path("/tmp/fake_archive/test_table.parquet"))

    @patch("subprocess.run")
    @patch("pathlib.Path.glob")
    def test_create_github_release(self, mock_glob, mock_run):
        # Mock finding parquet files
        mock_file1 = MagicMock()
        mock_file1.name = "users.parquet"
        mock_file2 = MagicMock()
        mock_file2.name = "orders.parquet"
        mock_glob.return_value = [mock_file1, mock_file2]
        
        # Setup run to return success for gh version and gh release create
        mock_success = MagicMock()
        mock_success.returncode = 0
        mock_run.return_value = mock_success

        self.archiver.create_github_release()

        # Should call gh --version
        self.assertEqual(mock_run.call_args_list[0][0][0], ['gh', '--version'])
        
        # Should call gh release create
        cmd = mock_run.call_args_list[1][0][0]
        self.assertEqual(cmd[0:3], ['gh', 'release', 'create'])
        self.assertIn('--title', cmd)
        self.assertIn('--notes', cmd)
        # Check files appended
        self.assertIn('users.parquet', cmd)
        self.assertIn('orders.parquet', cmd)

if __name__ == "__main__":
    unittest.main()
