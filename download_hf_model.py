#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face 模型下载脚本
支持从Hugging Face Hub下载模型，包括断点续传、镜像源配置等功能

使用示例:
    python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --cache_dir ./models
    python download_hf_model.py --model_name VLM2Vec/VLM2Vec-V2.0 --use_mirror --proxy http://127.0.0.1:7890
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("错误: 请先安装 huggingface_hub")
    print("运行: pip install huggingface_hub")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_hf_model.log')
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face 镜像源配置
MIRROR_ENDPOINTS = {
    'hf-mirror': 'https://hf-mirror.com',
    'modelscope': 'https://www.modelscope.cn/models',
    'official': 'https://huggingface.co'
}

class HFModelDownloader:
    """Hugging Face 模型下载器"""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 use_mirror: bool = False,
                 mirror_endpoint: str = 'hf-mirror',
                 proxy: Optional[str] = None,
                 token: Optional[str] = None):
        """
        初始化下载器
        
        Args:
            cache_dir: 模型缓存目录
            use_mirror: 是否使用镜像源
            mirror_endpoint: 镜像源类型
            proxy: 代理设置
            token: HuggingFace token
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self.use_mirror = use_mirror
        self.mirror_endpoint = mirror_endpoint
        self.proxy = proxy
        self.token = token
        
        # 创建缓存目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量
        self._setup_environment()
    
    def _setup_environment(self):
        """设置环境变量"""
        if self.use_mirror and self.mirror_endpoint in MIRROR_ENDPOINTS:
            endpoint = MIRROR_ENDPOINTS[self.mirror_endpoint]
            os.environ['HF_ENDPOINT'] = endpoint
            logger.info(f"使用镜像源: {endpoint}")
        
        if self.proxy:
            os.environ['HTTP_PROXY'] = self.proxy
            os.environ['HTTPS_PROXY'] = self.proxy
            logger.info(f"使用代理: {self.proxy}")
        
        if self.token:
            os.environ['HF_TOKEN'] = self.token
            logger.info("已设置 HuggingFace token")
    
    def download_model(self, 
                      model_name: str,
                      revision: str = "main",
                      ignore_patterns: Optional[List[str]] = None,
                      allow_patterns: Optional[List[str]] = None,
                      resume_download: bool = True) -> str:
        """
        下载完整模型
        
        Args:
            model_name: 模型名称，如 'VLM2Vec/VLM2Vec-V2.0'
            revision: 模型版本/分支
            ignore_patterns: 忽略的文件模式
            allow_patterns: 允许的文件模式
            resume_download: 是否支持断点续传
            
        Returns:
            str: 下载后的模型路径
        """
        logger.info(f"开始下载模型: {model_name}")
        logger.info(f"缓存目录: {self.cache_dir}")
        
        try:
            # 下载模型
            model_path = snapshot_download(
                repo_id=model_name,
                revision=revision,
                cache_dir=self.cache_dir,
                ignore_patterns=ignore_patterns,
                allow_patterns=allow_patterns,
                resume_download=resume_download,
                token=self.token
            )
            
            logger.info(f"模型下载成功: {model_path}")
            return model_path
            
        except HfHubHTTPError as e:
            logger.error(f"HTTP错误: {e}")
            if "401" in str(e):
                logger.error("认证失败，请检查token是否正确")
            elif "404" in str(e):
                logger.error(f"模型 {model_name} 不存在或无权限访问")
            raise
        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise
    
    def download_file(self, 
                     model_name: str,
                     filename: str,
                     revision: str = "main",
                     subfolder: Optional[str] = None) -> str:
        """
        下载单个文件
        
        Args:
            model_name: 模型名称
            filename: 文件名
            revision: 模型版本
            subfolder: 子文件夹
            
        Returns:
            str: 下载后的文件路径
        """
        logger.info(f"下载文件: {model_name}/{filename}")
        
        try:
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                revision=revision,
                subfolder=subfolder,
                cache_dir=self.cache_dir,
                token=self.token
            )
            
            logger.info(f"文件下载成功: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            raise
    
    def check_model_exists(self, model_name: str) -> bool:
        """
        检查模型是否已存在于本地缓存
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否存在
        """
        # 简化的检查逻辑，实际可能需要更复杂的验证
        model_cache_path = Path(self.cache_dir) / f"models--{model_name.replace('/', '--')}"
        exists = model_cache_path.exists()
        logger.info(f"模型 {model_name} 本地缓存{'存在' if exists else '不存在'}")
        return exists

def main():
    parser = argparse.ArgumentParser(description="Hugging Face 模型下载工具")
    parser.add_argument("--model_name", 
                       required=True,
                       help="模型名称，如 VLM2Vec/VLM2Vec-V2.0")
    parser.add_argument("--cache_dir", 
                       default=None,
                       help="模型缓存目录 (默认: ~/.cache/huggingface/hub)")
    parser.add_argument("--revision", 
                       default="main",
                       help="模型版本/分支 (默认: main)")
    parser.add_argument("--use_mirror", 
                       action="store_true",
                       help="使用镜像源下载")
    parser.add_argument("--mirror_endpoint", 
                       choices=list(MIRROR_ENDPOINTS.keys()),
                       default="hf-mirror",
                       help="镜像源类型")
    parser.add_argument("--proxy", 
                       help="代理设置，如 http://127.0.0.1:7890")
    parser.add_argument("--token", 
                       help="HuggingFace token")
    parser.add_argument("--file_only", 
                       help="仅下载指定文件")
    parser.add_argument("--ignore_patterns", 
                       nargs="*",
                       help="忽略的文件模式")
    parser.add_argument("--allow_patterns", 
                       nargs="*",
                       help="允许的文件模式")
    parser.add_argument("--check_only", 
                       action="store_true",
                       help="仅检查模型是否存在")
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = HFModelDownloader(
        cache_dir=args.cache_dir,
        use_mirror=args.use_mirror,
        mirror_endpoint=args.mirror_endpoint,
        proxy=args.proxy,
        token=args.token
    )
    
    try:
        if args.check_only:
            # 仅检查模型是否存在
            exists = downloader.check_model_exists(args.model_name)
            print(f"模型 {args.model_name} {'已存在' if exists else '不存在'}")
            return
        
        if args.file_only:
            # 下载单个文件
            file_path = downloader.download_file(
                model_name=args.model_name,
                filename=args.file_only,
                revision=args.revision
            )
            print(f"文件下载完成: {file_path}")
        else:
            # 下载完整模型
            model_path = downloader.download_model(
                model_name=args.model_name,
                revision=args.revision,
                ignore_patterns=args.ignore_patterns,
                allow_patterns=args.allow_patterns
            )
            print(f"模型下载完成: {model_path}")
            
    except KeyboardInterrupt:
        logger.info("用户中断下载")
        sys.exit(1)
    except Exception as e:
        logger.error(f"下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()