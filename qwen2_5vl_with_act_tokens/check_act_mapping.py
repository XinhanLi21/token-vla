import os, json
import numpy as np
from transformers import AutoProcessor

save_dir = "./"  # 你就在该目录执行
K = 2048
codebook_path = os.path.join(save_dir, "codebook_uniform_2048.npy")

processor = AutoProcessor.from_pretrained(save_dir, trust_remote_code=True, local_files_only=True)
tokenizer = processor.tokenizer

codebook = np.load(codebook_path)
assert codebook.shape[0] == K, f"codebook K不对: {codebook.shape}"

act_strs = [f"<act_{i:04d}>" for i in range(K)]
act_ids = tokenizer.convert_tokens_to_ids(act_strs)

unk = tokenizer.unk_token_id
bad = [i for i, tid in enumerate(act_ids) if tid == unk]
assert len(bad) == 0, f"这些动作token变成了unk，说明没加进词表: {bad[:20]}..."

# 连续性检查（推荐用连续offset映射）
is_contiguous = (act_ids == list(range(act_ids[0], act_ids[0] + K)))
print("act_id_contiguous:", is_contiguous)
print("act_base_id:", act_ids[0], "act_last_id:", act_ids[-1])

# 保存 meta（强烈建议）
meta = {
    "K": K,
    "token_format": "<act_{i:04d}>",
    "codebook_file": "codebook_uniform_2048.npy",
    "act_ids_contiguous": bool(is_contiguous),
    "act_base_id": int(act_ids[0]) if is_contiguous else None,
}

with open(os.path.join(save_dir, "action_token_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("saved action_token_meta.json")

# 自检：随机挑几个idx验证 tok -> id -> idx -> codebook 行
test = [0, 1, 7, 123, 2047]
for idx in test:
    tok = f"<act_{idx:04d}>"
    tid = tokenizer.convert_tokens_to_ids(tok)
    if is_contiguous:
        assert tid == act_ids[0] + idx
        back_idx = tid - act_ids[0]
    else:
        # 非连续情况，保底用字符串解析回 idx
        back_idx = int(tok[len("<act_"):len("<act_")+4])
    assert back_idx == idx
    _ = codebook[idx]
print("OK: mapping consistent.")
