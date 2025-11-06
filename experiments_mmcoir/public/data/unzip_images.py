import argparse
import os
import tarfile
from typing import Iterator, Tuple


def repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "../../.."))


def iter_image_archives(base_dir: str, include_tar: bool = True) -> Iterator[Tuple[str, str, str]]:
    """遍历 base_dir 下所有子目录，查找 images.tar.gz（以及可选 images.tar）。

    返回 (root_dir, archive_path, target_images_dir)
    """
    for root, dirs, files in os.walk(base_dir):
        if "images.tar.gz" in files:
            archive = os.path.join(root, "images.tar.gz")
            yield root, archive, os.path.join(root, "images")
        if include_tar and "images.tar" in files:
            archive = os.path.join(root, "images.tar")
            yield root, archive, os.path.join(root, "images")


def _is_within_directory(directory: str, target: str) -> bool:
    directory = os.path.abspath(directory)
    target = os.path.abspath(target)
    try:
        common = os.path.commonpath([directory, target])
    except ValueError:
        # On Windows, mixing drive letters can raise ValueError
        return False
    return common == directory


def safe_extract(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"阻止潜在的路径穿越: {member.name}")
    tar.extractall(path)


def extract_archive(archive_path: str, target_dir: str, overwrite: bool) -> None:
    if os.path.isdir(target_dir) and not overwrite:
        print(f"跳过：目标已存在且未指定 --overwrite -> {target_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)
    mode = "r:*"  # 支持 .tar.gz / .tar
    with tarfile.open(archive_path, mode) as tar:
        print(f"解压 {archive_path} 到 {target_dir}")
        safe_extract(tar, target_dir)
    print(f"完成解压: {archive_path}")


def main():
    parser = argparse.ArgumentParser(description="遍历并解压 images.tar.gz 到 images 文件夹")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(repo_root(), "data"),
        help="下载后的根目录，默认是 <repo_root>/data"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 images 目录已存在，允许覆盖"
    )
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="解压成功后删除归档文件 (images.tar.gz / images.tar)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将要解压的归档，不执行"
    )
    parser.add_argument(
        "--include-tar",
        action="store_true",
        help="除了 images.tar.gz 外，也处理 images.tar"
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"扫描根目录: {base_dir}")
    found_any = False
    for root, archive_path, target_dir in iter_image_archives(base_dir, include_tar=args.include_tar):
        found_any = True
        print(f"发现归档: {archive_path}")
        print(f"目标目录: {target_dir}")
        if args.dry_run:
            continue
        try:
            extract_archive(archive_path, target_dir, overwrite=args.overwrite)
            if args.delete_archive:
                try:
                    os.remove(archive_path)
                    print(f"已删除归档: {archive_path}")
                except OSError as e:
                    print(f"删除归档失败: {archive_path} -> {e}")
        except Exception as e:
            print(f"解压失败: {archive_path} -> {e}")

    if not found_any:
        print("未在指定目录下发现 images.tar.gz 或 images.tar")
    else:
        if args.dry_run:
            print("dry-run 预览完成：未执行任何解压")
        else:
            print("处理完成")


if __name__ == "__main__":
    main()