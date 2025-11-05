import os
import argparse
from huggingface_hub import HfApi

def upload_file(local_file_path: str, repo_id: str, remote_path: str, commit_message: str = "Upload file"):
    """
    Upload a local file to a Hugging Face dataset repository.

    Args:
        local_file_path (str): Path to the local file to upload.
        repo_id (str): Hugging Face repository ID (e.g., "username/repo_name").
        remote_path (str): Path in the remote repository where the file will be uploaded.
        commit_message (str): Commit message for the upload.

    Raises:
        FileNotFoundError: If the local file does not exist.
        Exception: If the upload fails.
    """
    if not os.path.isfile(local_file_path):
        raise FileNotFoundError(f"Local file not found: {local_file_path}")

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_file_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=remote_path,
            commit_message=commit_message,
        )
        print(f"[OK] File uploaded: {local_file_path} -> {repo_id}:{remote_path}")
    except Exception as e:
        print(f"[ERROR] Failed to upload file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Hugging Face dataset repository")
    parser.add_argument("--local-file", default="MMCoIR-test/README.md", help="Path to the local file to upload")
    parser.add_argument("--repo-id", default="JiahuiGengNLP/MMCoIR-test", help="Hugging Face repository ID")
    parser.add_argument("--remote-path", default="README.md", help="Path in the remote repository")
    parser.add_argument("--commit-message", default="Upload README.md", help="Commit message for the upload")

    args = parser.parse_args()

    upload_file(
        local_file_path=args.local_file,
        repo_id=args.repo_id,
        remote_path=args.remote_path,
        commit_message=args.commit_message,
    )