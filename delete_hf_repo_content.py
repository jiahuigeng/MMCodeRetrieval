import argparse
import os
from typing import List
from huggingface_hub import HfApi

# 提示：如果你已经通过 `huggingface-cli login` 登录，API 将自动使用缓存令牌。
# 若你希望显式覆盖，也可设置环境变量 `HF_TOKEN`，本脚本会优先使用该变量。

api = HfApi()

def _get_token() -> str | None:
    """优先使用环境变量中的令牌，否则让 SDK 使用本地登录缓存。"""
    return os.getenv("HF_TOKEN") or None

def _list_files(repo_id: str, repo_type: str, token: str | None) -> List[str]:
    return api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)

def delete_hf_content(
    repo_id: str,
    path: str,
    is_folder: bool,
    repo_type: str = "dataset",
    confirm: bool = False,
    non_recursive: bool = False,
) -> None:
    """删除 Hugging Face 仓库中的文件或文件夹。

    repo_type 可选："dataset"（默认）、"model"、"space"。
    """
    token = _get_token()

    print(f"列出仓库 {repo_id} (type={repo_type}) 中的文件...")
    files = _list_files(repo_id=repo_id, repo_type=repo_type, token=token)
    if not files:
        print("未获取到文件列表，检查仓库是否存在，以及登录/权限是否正常。")

    # 规范化匹配前缀，仅匹配 'ChartGen/' 下的内容，避免误删 'ChartGen_c2i/'
    normalized_path = path.strip("/")
    folder_prefix = normalized_path + "/"

    if is_folder:
        # 仅删除目录下文件：严格以 'ChartGen/' 为前缀，不再使用裸 'ChartGen' 前缀
        files_to_delete = [f for f in files if f.startswith(folder_prefix)]
        if non_recursive:
            # 仅删除顶层文件：过滤掉子目录中的文件
            files_to_delete = [
                f for f in files_to_delete
                if "/" not in f[len(folder_prefix):]
            ]
        if not files_to_delete:
            print(f"未找到文件夹 {path} 下的文件，可能文件夹不存在或已为空。")
            return
        print(f"将删除 {len(files_to_delete)} 个条目：")
        for fp in files_to_delete:
            print(f"  - {fp}")
        if not confirm:
            print("未提供 --confirm，当前仅为预览。未执行删除。")
            return
        for file_path in files_to_delete:
            print(f"删除: {file_path}")
            api.delete_file(
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=f"删除 {normalized_path} 下的文件 {file_path}"
            )
    else:
        # 删除单个文件：要求精确匹配路径
        if normalized_path not in files:
            print(f"文件 {path} 不存在。")
            return
        print(f"将删除文件：{normalized_path}")
        if not confirm:
            print("未提供 --confirm，当前仅为预览。未执行删除。")
            return
        print(f"删除: {normalized_path}")
        api.delete_file(
            path_in_repo=normalized_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"删除文件 {normalized_path}"
        )

    print("删除操作完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除 Hugging Face 仓库中的文件或文件夹")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face 仓库 ID，例如 JiahuiGengNLP/MMCoIR-train")
    parser.add_argument("--path", type=str, required=True, help="要删除的文件或文件夹路径，例如 ChartGen 或 ChartGen/file.txt")
    parser.add_argument("--is-folder", action="store_true", help="是否删除整个文件夹")
    parser.add_argument("--repo-type", type=str, default="dataset", choices=["dataset", "model", "space"], help="仓库类型，默认为 dataset")
    parser.add_argument("--confirm", action="store_true", help="确认执行删除。未提供则仅预览")
    parser.add_argument("--non-recursive", action="store_true", help="仅删除目录下的顶层文件，不递归子目录")

    args = parser.parse_args()

    delete_hf_content(
        args.repo_id,
        args.path,
        args.is_folder,
        repo_type=args.repo_type,
        confirm=args.confirm,
        non_recursive=args.non_recursive,
    )