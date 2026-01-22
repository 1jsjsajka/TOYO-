import pandas as pd
import os  
from sklearn.model_selection import train_test_split

input_file = r'C:\Users\zhuju\OneDrive\文档\卒研\LLM合并データ\3models一致.xlsx'
output_dir = r'C:\Users\zhuju\OneDrive\文档\卒研\LLM人工标注'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_excel(input_file)


train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

print(f"训练集数量: {len(train_df)}")
print(f"全量测试集数量: {len(test_df)}")

# 从 1600 条中随机抽取 400 条用于人工标注
manual_test_df = test_df.sample(n=400, random_state=7)

# 准备人工标注用的表格
text_column_name = 'text'  
annotation_df = manual_test_df[[text_column_name]].copy()
annotation_df['human_label'] = "" 

# 保存
train_df.to_excel(os.path.join(output_dir, 'train_data_4400_with_llm_labels.xlsx'), index=False)
test_df.to_excel(os.path.join(output_dir, 'test_data_full_with_llm_labels.xlsx'), index=False)
manual_test_df.to_excel(os.path.join(output_dir, 'manual_test_400_with_llm_labels.xlsx'), index=False)
annotation_df.to_excel(os.path.join(output_dir, 'for_manual_annotation_400.xlsx'), index=False)

print("\n--- 运行成功！ ---")
print(f"文件夹中现在已生成以下 4 个文件：")
print(f"1. train_data_4400_with_llm_labels.xlsx (训练用)")
print(f"2. test_data_full_1600_with_llm_labels.xlsx (大样本测试用)")
print(f"3. for_manual_annotation_400.xlsx (人工打标用)")
print(f"4. manual_test_400_with_llm_labels.xlsx (人工打标对照用)")