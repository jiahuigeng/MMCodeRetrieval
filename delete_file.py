from huggingface_hub import HfApi

api = HfApi()
repo_id = "JiahuiGengNLP/MMCoIR-train"   # 或你的目标数据集仓库

api.delete_file(
    path_in_repo="dataset_script.py",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Remove dataset_script.py"
)


repo_id = "JiahuiGengNLP/MMCoIR-test"   # 或你的目标数据集仓库

api.delete_file(
    path_in_repo="dataset_script.py",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Remove dataset_script.py"
)