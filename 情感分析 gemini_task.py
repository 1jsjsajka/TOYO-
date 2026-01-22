import pandas as pd
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# ================= 配置区域 =================
API_KEY = "" # Key

INPUT_FILE = r"C:\Users\zhuju\OneDrive\文档\卒研\LLM标签データ\最終データgemini.xlsx" 
OUTPUT_FILE = r"C:\Users\zhuju\OneDrive\文档\卒研\LLM标签データ\最終データgemini_result.xlsx"


MAX_WORKERS = 20 


genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3-flash-preview')

def get_label(idx, text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return idx, "中立"
    
    prompt = f"对婚姻观文本进行情感分类。只能输出：正面、中性、负面。内容：{text}"
    
    for _ in range(2):
        try:
            response = model.generate_content(prompt)
            res = response.text.strip()
            if "正" in res: return idx, "正面"
            if "负" in res: return idx, "负面"
            return idx, "中性"
        except:
            time.sleep(0.5)
    return idx, "API请求失败"

def main():
    print("1. 正在尝试读取 Excel 文件，请确保文件没有被 Excel 软件打开...")
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"   - 成功读取文件，共 {len(df)} 行。")
    except Exception as e:
        print(f"   - [错误] 读取失败: {e}")
        return

    
    if '情感标签' not in df.columns:
        df['情感标签'] = None
    

    mask = (df['情感标签'] == "API请求失败") | (df['情感标签'].isna())
    to_process = df[mask]
    
    print(f"2. 待补齐行数: {len(to_process)}")

    if len(to_process) == 0:
        print("   - 无需补齐，所有数据已处理完毕。")
        return

    print("3. 正在启动多线程加速分析...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
     
        futures = {executor.submit(get_label, i, row.iloc[3]): i for i, row in to_process.iterrows()}
        
        count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="分析进度"):
            idx, label = future.result()
            df.at[idx, '情感标签'] = label
            count += 1
            
           
            if count % 100 == 0:
                df.to_excel(OUTPUT_FILE, index=False)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n--- 任务全部完成 ---")
    print(f"最终结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()