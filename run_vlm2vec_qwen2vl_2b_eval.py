#!/usr/bin/env python3
"""
VLM2Vec-Qwen2VL-2B 评估脚本

这个脚本演示如何运行VLM2Vec-Qwen2VL-2B模型的评估功能。
支持图像、视频和视觉文档三种模态的评估。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def setup_environment():
    """设置环境变量和路径"""
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 设置PYTHONPATH环境变量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in current_pythonpath:
        os.environ['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
    
    print(f"✅ 项目根目录: {project_root}")
    print(f"✅ Python路径已设置")

def run_evaluation(model_name, model_backbone, modality, data_basedir, output_basedir, batch_size=8, gpu_ids="0"):
    """运行VLM2Vec评估
    
    Args:
        model_name: 模型名称，如 'VLM2Vec/VLM2Vec-V2.0'
        model_backbone: 模型骨干网络，如 'qwen2_vl'
        modality: 评估模态，可选 'image', 'video', 'visdoc'
        data_basedir: 数据集根目录
        output_basedir: 输出结果根目录
        batch_size: 批处理大小
        gpu_ids: GPU设备ID
    """
    
    # 配置文件路径
    config_path = f"experiments/public/eval/{modality}.yaml"
    output_path = f"{output_basedir}/{model_name.split('/')[-1]}/{modality}/"
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 构建评估命令
    cmd = [
        "python", "eval.py",
        "--pooling", "eos",
        "--normalize", "true",
        "--per_device_eval_batch_size", str(batch_size),
        "--model_backbone", model_backbone,
        "--model_name", model_name,
        "--dataset_config", config_path,
        "--encode_output_path", output_path,
        "--data_basedir", data_basedir
    ]
    
    # 设置CUDA设备
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    print(f"\n🚀 开始评估 {model_name} - {modality} 模态")
    print(f"📁 配置文件: {config_path}")
    print(f"📁 输出路径: {output_path}")
    print(f"🔧 批处理大小: {batch_size}")
    print(f"🎯 GPU设备: {gpu_ids}")
    print(f"\n执行命令: {' '.join(cmd)}")
    
    try:
        # 运行评估
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ {modality} 模态评估完成")
            print("\n📊 评估输出:")
            print(result.stdout)
        else:
            print(f"❌ {modality} 模态评估失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行评估时发生错误: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='VLM2Vec-Qwen2VL-2B 评估脚本')
    
    # 模型配置
    parser.add_argument('--model_name', default='VLM2Vec/VLM2Vec-V2.0', 
                       help='模型名称 (默认: VLM2Vec/VLM2Vec-V2.0)')
    parser.add_argument('--model_backbone', default='qwen2_vl',
                       help='模型骨干网络 (默认: qwen2_vl)')
    
    # 评估配置
    parser.add_argument('--modality', choices=['image', 'video', 'visdoc', 'all'], 
                       default='image', help='评估模态 (默认: image)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批处理大小 (默认: 8)')
    parser.add_argument('--gpu_ids', default='0',
                       help='GPU设备ID (默认: 0)')
    
    # 路径配置
    parser.add_argument('--data_basedir', default='~/data/vlm2vec_eval',
                       help='数据集根目录 (默认: ~/data/vlm2vec_eval)')
    parser.add_argument('--output_basedir', default='./eval_results',
                       help='输出结果根目录 (默认: ./eval_results)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 VLM2Vec-Qwen2VL-2B 评估脚本")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 展开用户目录路径
    data_basedir = os.path.expanduser(args.data_basedir)
    output_basedir = os.path.expanduser(args.output_basedir)
    
    print(f"\n📋 评估配置:")
    print(f"  模型: {args.model_name}")
    print(f"  骨干网络: {args.model_backbone}")
    print(f"  评估模态: {args.modality}")
    print(f"  数据目录: {data_basedir}")
    print(f"  输出目录: {output_basedir}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  GPU设备: {args.gpu_ids}")
    
    # 检查数据目录
    if not os.path.exists(data_basedir):
        print(f"\n⚠️  警告: 数据目录不存在: {data_basedir}")
        print("请确保已下载评估数据集，或修改 --data_basedir 参数")
        print("\n💡 提示: 可以使用以下命令下载数据:")
        print("bash experiments/public/data/download_data.sh")
    
    # 运行评估
    success_count = 0
    total_count = 0
    
    if args.modality == 'all':
        modalities = ['image', 'video', 'visdoc']
    else:
        modalities = [args.modality]
    
    for modality in modalities:
        total_count += 1
        if run_evaluation(
            model_name=args.model_name,
            model_backbone=args.model_backbone,
            modality=modality,
            data_basedir=data_basedir,
            output_basedir=output_basedir,
            batch_size=args.batch_size,
            gpu_ids=args.gpu_ids
        ):
            success_count += 1
    
    print(f"\n=" * 60)
    print(f"📊 评估完成: {success_count}/{total_count} 个模态评估成功")
    
    if success_count == total_count:
        print("🎉 所有评估任务完成!")
        print(f"📁 结果保存在: {output_basedir}")
    else:
        print("⚠️  部分评估任务失败，请检查错误信息")
    
    print("=" * 60)

if __name__ == '__main__':
    main()