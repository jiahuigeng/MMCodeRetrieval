import argparse
import os
import tarfile
import shutil
from typing import Iterator, Tuple


def repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "../../.."))


def iter_image_archives(base_dir: str, include_tar: bool = True) -> Iterator[Tuple[str, str, str]]:
    """遍历 base_dir 下所有子目录，查找 images.tar.gz（以及可选 images.tar）。

    返回 (root_dir, archive_path, target_images_dir)
    注意：如果归档位于某个 'images' 目录内，则目标 images 目录应为该目录（避免出现 images/images）。
    """
    for root, dirs, files in os.walk(base_dir):
        if "images.tar.gz" in files:
            archive = os.path.join(root, "images.tar.gz")
            target = root if os.path.basename(root).lower() == "images" else os.path.join(root, "images")
            yield root, archive, target
        if include_tar and "images.tar" in files:
            archive = os.path.join(root, "images.tar")
            target = root if os.path.basename(root).lower() == "images" else os.path.join(root, "images")
            yield root, archive, target


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
    """解压归档到正确位置并避免产生 images/images。

    规则：
    - 若归档内部包含顶层 `images/`，则应解压到目标 images 的父目录，以得到 `<...>/images/...`。
    - 若归档位于某个 `.../images/` 目录中，则应解压到其父目录（从而避免出现 `images/images`）。
    - 支持 overwrite：存在目标 images 目录时，允许强制覆盖（删除后再解压）。
    """

    # 归档所在目录与是否位于 images 下
    archive_dir = os.path.dirname(archive_path)
    archive_dir_is_images = os.path.basename(archive_dir).lower() == "images"

    # 目标 images 目录（用于覆盖与提示）
    target_images_dir = target_dir if os.path.basename(target_dir).lower() == "images" else os.path.join(target_dir, "images")

    # 如果存在且不覆盖，则跳过
    if os.path.isdir(target_images_dir) and not overwrite:
        print(f"跳过：目标已存在且未指定 --overwrite -> {target_images_dir}")
        return

    mode = "r:*"  # 支持 .tar.gz / .tar
    with tarfile.open(archive_path, mode) as tar:
        members = tar.getmembers()

        # 判断归档内部是否以 `images/` 为顶层目录（由打包脚本使用 arcname="images" 产生）
        has_images_root = any(m.name == "images" and m.isdir() for m in members)
        all_under_images = all(m.name == "images" or m.name.startswith("images/") for m in members)

        # 计算解压路径：
        # - 若内部含顶层 images，解压到应出现的 images 父目录：
        #   · 若归档位于 .../images 目录内，则取其父目录
        #   · 否则解压到归档所在目录
        # - 若不含顶层 images，则解压到 target_dir（与原逻辑一致）
        if has_images_root or all_under_images:
            extract_path = os.path.dirname(archive_dir) if archive_dir_is_images else archive_dir
        else:
            extract_path = target_dir

        # 覆盖：提前删除目标 images 目录，避免残留导致 images/images 或旧文件
        # 最终 images 应出现在 extract_path/images
        final_images_dir = os.path.join(extract_path, "images")
        if overwrite and os.path.isdir(final_images_dir):
            print(f"覆盖模式：删除已存在的目录 -> {final_images_dir}")
            shutil.rmtree(final_images_dir)

        os.makedirs(extract_path, exist_ok=True)
        print(f"解压 {archive_path} 到 {extract_path}")
        safe_extract(tar, extract_path)
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
            # 对于 train 集强制覆盖：当路径包含 'MMCoIR_train'（不区分大小写）时，忽略现有目录直接覆盖
            is_train = ("mmcoir_train" in base_dir.lower()) or ("mmcoir_train" in root.lower())
            overwrite = args.overwrite or is_train
            extract_archive(archive_path, target_dir, overwrite=overwrite)
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