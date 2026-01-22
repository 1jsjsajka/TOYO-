import pandas as pd
from collections import Counter

# 配置：你的三个result文件 
GEMINI_FILE = r"c:\Users\zhuju\OneDrive\文档\卒研\データ\最終データgemini_result.xlsx"
QWEN_FILE   = r"c:\Users\zhuju\OneDrive\文档\卒研\データ\最終データqwen_result.xlsx"
DEEP_FILE   = r"c:\Users\zhuju\OneDrive\文档\卒研\データ\最終データdeepseek_result.xlsx"


#读取文本 标签

TEXT_COL_IDX = 3   
LABEL_COL_IDX = 4  

def load_result(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    out = df.iloc[:, [TEXT_COL_IDX, LABEL_COL_IDX]].copy()
    out.columns = ["text", model_name]

    
    out["text"] = out["text"].astype(str)
    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[out["text"].notna() & (out["text"] != "")]

   
    out[model_name] = out[model_name].astype(str).str.strip()

    
    out = out.drop_duplicates(subset=["text"], keep="first")

    return out

gem = load_result(GEMINI_FILE, "gemini")
qwe = load_result(QWEN_FILE, "qwen")
dep = load_result(DEEP_FILE, "deepseek")

#合并
merged = gem.merge(qwe, on="text", how="outer").merge(dep, on="text", how="outer")

print("合并后总行数:", len(merged))
print("三家都有标签的行数:", merged.dropna(subset=["gemini", "qwen", "deepseek"]).shape[0])

# 统一标签
label_map = {
    "正面": "pos", "正向": "pos", "积极": "pos", "pos": "pos", "positive": "pos",
    "中性": "neu", "中立": "neu", "neu": "neu", "neutral": "neu",
    "负面": "neg", "负向": "neg", "消极": "neg", "neg": "neg", "negative": "neg",
}

def norm_label(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return label_map.get(x, x)

for c in ["gemini", "qwen", "deepseek"]:
    merged[c] = merged[c].apply(norm_label)

# 多数投票 + 一致性
def vote(row):
    labels = [row["gemini"], row["qwen"], row["deepseek"]]
    if any(x is None for x in labels):
        return pd.Series({"final_label": None, "agreement": "missing"})

    cnt = Counter(labels)
    top_label, top_n = cnt.most_common(1)[0]

    if len(cnt) == 1:
        agreement = "3of3"
    elif top_n == 2:
        agreement = "2of3"
    else:
        agreement = "0of3"

    return pd.Series({"final_label": top_label, "agreement": agreement})

merged[["final_label", "agreement"]] = merged.apply(vote, axis=1)

# 一致性统计
stat = merged["agreement"].value_counts(dropna=False)
stat_ratio = (stat / len(merged)).round(4)
stat_table = pd.DataFrame({"count": stat, "ratio": stat_ratio}).reset_index().rename(columns={"index":"agreement"})

print("\n=== 一致性统计 ===")
print(stat_table)

#导出
import os

OUTPUT_DIR = r"C:\Users\zhuju\OneDrive\文档\卒研\LLM合并データ"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 合并后的总表
merged.to_excel(
    os.path.join(OUTPUT_DIR, "merged_3models_with_vote.xlsx"),
    index=False
)

# 一致性统计表
stat_table.to_csv(
    os.path.join(OUTPUT_DIR, "agreement_stats.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 3/3 一致
train_high = merged[merged["agreement"] == "3of3"][["text", "final_label"]].dropna()
train_high.to_csv(
    os.path.join(OUTPUT_DIR, "train_high_confidence.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 3/3 + 2/3
train_mid = merged[merged["agreement"].isin(["3of3", "2of3"])][["text", "final_label"]].dropna()
train_mid.to_csv(
    os.path.join(OUTPUT_DIR, "train_mid_confidence.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 0/3
conflicts = merged[merged["agreement"] == "0of3"][["text", "gemini", "qwen", "deepseek"]].dropna(subset=["text"])
conflicts.to_excel(
    os.path.join(OUTPUT_DIR, "conflicts_0of3.xlsx"),
    index=False
)

print(f"\n✅ 已全部导出到目录：{OUTPUT_DIR}")
