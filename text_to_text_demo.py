from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, Qwen2_VL_process_fn
from src.utils import batch_to_device
import torch

# 配置模型参数
model_args = ModelArguments(
    model_name='VLM2Vec/VLM2Vec-V2.0',
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)
data_args = DataArguments()

# 加载处理器和模型
processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args)
model = model.to('cuda', dtype=torch.bfloat16)
model.eval()

print("=== VLM2Vec 文本到文本检索示例 ===")

# 示例1：单个文本查询和目标
print("\n1. 单个文本查询示例:")
query_text = "What is artificial intelligence?"
target_text = "Artificial intelligence is a branch of computer science that aims to create intelligent machines."

# 处理查询文本
query_inputs = processor(text=query_text, images=None, return_tensors="pt")
query_inputs = {key: value.to('cuda') for key, value in query_inputs.items()}
query_output = model(qry=query_inputs)["qry_reps"]

# 处理目标文本
target_inputs = processor(text=target_text, images=None, return_tensors="pt")
target_inputs = {key: value.to('cuda') for key, value in target_inputs.items()}
target_output = model(tgt=target_inputs)["tgt_reps"]

# 计算相似度
similarity = model.compute_similarity(query_output, target_output)
print(f"查询: {query_text}")
print(f"目标: {target_text}")
print(f"相似度: {similarity.item():.4f}")

# 示例2：批量文本检索
print("\n2. 批量文本检索示例:")
queries = [
    "How does machine learning work?",
    "What is deep learning?"
]

targets = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
    "Python is a programming language widely used in data science.",
    "Natural language processing deals with the interaction between computers and human language."
]

# 批量处理查询
processor_inputs = {
    "text": queries,
    "images": [None] * len(queries),
}
query_inputs = Qwen2_VL_process_fn(processor_inputs, processor)
query_inputs = batch_to_device(query_inputs, "cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    query_embeddings = model(qry=query_inputs)["qry_reps"]

# 批量处理目标
processor_inputs = {
    "text": targets,
    "images": [None] * len(targets),
}
target_inputs = Qwen2_VL_process_fn(processor_inputs, processor)
target_inputs = batch_to_device(target_inputs, "cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    target_embeddings = model(tgt=target_inputs)["tgt_reps"]

# 计算所有查询与所有目标的相似度
similarity_matrix = model.compute_similarity(query_embeddings, target_embeddings)

print("\n相似度矩阵 (查询 x 目标):")
for i, query in enumerate(queries):
    print(f"\n查询 {i+1}: {query}")
    for j, target in enumerate(targets):
        print(f"  目标 {j+1}: {similarity_matrix[i][j].item():.4f} - {target[:50]}...")
    # 找到最相似的目标
    best_match_idx = torch.argmax(similarity_matrix[i]).item()
    print(f"  最佳匹配: 目标 {best_match_idx+1} (相似度: {similarity_matrix[i][best_match_idx].item():.4f})")

print("\n=== 文本到文本检索测试完成 ===")