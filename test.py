import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

# --- 0. 配置参数 ---
# 你的模型保存路径应该和训练脚本中保存 LoRA 适配器的路径一致
LORA_ADAPTER_DIR = "./qwen_finetuned_hate_speech_v4_qwen3/final_lora_adapters"  # <-- 更改为 Qwen3 的路径
BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"  # <-- 更改为 Qwen3 模型名称
TEST_FILE = 'test1.json'
OUTPUT_TXT_FILE = 'output_full_dataset_formatted_qwen3.txt'  # <-- 更改输出文件名以区分 Qwen3
# MAX_SEQ_LENGTH 保持与训练时一致
MAX_SEQ_LENGTH = 512

# 定义回退值（与你训练代码中的 DEFAULT_TARGET_GROUP_FALLBACK 和 DEFAULT_HAS_HATE_FALLBACK 保持一致）
DEFAULT_TARGET_GROUP_FALLBACK = "未知群体"
DEFAULT_HAS_HATE_FALLBACK = "未知"

# !!! 核心点：请在这里手动填入你在训练脚本运行后，控制台输出中
# !!! “--- 从数据集中动态收集到的所有唯一 '目标群体' 标签 ---”
# !!! 和 “--- 从数据集中动态收集到的所有唯一 '是否具有仇恨' 标签 ---”
# !!! 这两部分下方列出的所有标签。
# !!! 这些标签必须与模型训练时看到的完全一致，否则规范化会出问题。

# 示例标签（请务必替换为你的实际标签！）
# 你可以从你的训练日志中找到类似以下的输出：
# --- 从数据集中动态收集到的所有唯一 '目标群体' 标签 ---
# - 少数民族
# - 女性
# - ...
# --- 从数据集中动态收集到的所有唯一 '是否具有仇恨' 标签 ---
# - 是
# - 否
COLLECTED_TARGETED_GROUPS = {
    "Sexism", "LGBTQ", "Racism", "Region", "others", "non-hate", # <-- 替换为你的实际标签集合
}
COLLECTED_HAS_HATE_LABELS = {
    "hate", "non-hate"  # <-- 替换为你的实际标签集合
}
# !!! ------------------------------------------------------------------

# 确保CUDA可用
if not torch.cuda.is_available():
    raise SystemError("CUDA 不可用。请确保你有兼容的 GPU 并已安装 CUDA。")
print(f"正在使用设备: {torch.cuda.current_device()}")


# --- 1. 定义规范化逻辑 (与训练脚本中的 normalize_label 函数一致) ---
def normalize_label(label_str, field_name, allowed_labels_set=None, fallback_value=None):
    """
    将标签字符串规范化。
    对于 '目标群体' 和 '是否具有仇恨' 字段，将其规范化为允许的标签集中的一个。
    对于其他字段 (评论对象, 论点)，只进行基本清理，不进行标签匹配。
    """
    label_str = label_str.strip()

    if field_name in ['targeted_group', 'has_hate']:
        if allowed_labels_set is None or fallback_value is None:
            raise ValueError(f"对于 '{field_name}'，必须提供 allowed_labels_set 和 fallback_value。")

        if label_str in allowed_labels_set:
            return label_str
        else:
            if field_name == 'has_hate':
                lower_label = label_str.lower()
                # 尝试匹配常见的是/否变体
                if "是" in allowed_labels_set and (
                        lower_label == '是' or lower_label in {"yes", "true", "有", "hate", "hateful"}):
                    return "是"
                if "否" in allowed_labels_set and (
                        lower_label == '否' or lower_label in {"no", "false", "not hate", "无", "中性", "neutral"}):
                    return "否"
            return fallback_value
    else:
        # 对于自由文本字段 (评论对象, 论点)，只进行基本清理
        if label_str.lower() == 'null' or not label_str:  # 如果是 'null' 或空字符串，则返回 '无'
            return "无"
        return label_str


# 辅助函数，用于解析模型生成的输出并进行规范化
def parse_and_format_generated_output(generated_text, targeted_groups_set, has_hate_labels_set):
    """
    尝试从模型生成的文本中解析出四元组列表，并对标签进行规范化。
    """
    formatted_quads = []
    segments = generated_text.split('[SEP]')
    for seg in segments:
        seg = seg.replace('[END]', '').strip()
        parts = [p.strip() for p in seg.split('|')]

        # 期望模型输出 4 部分
        if len(parts) == 4:
            target_raw, argument_raw, targeted_group_raw, has_hate_raw = parts

            # 规范化模型生成的标签，使用手动定义的标签集合
            target = normalize_label(target_raw, 'target')
            argument = normalize_label(argument_raw, 'argument')
            targeted_group = normalize_label(targeted_group_raw, 'targeted_group',
                                             targeted_groups_set, DEFAULT_TARGET_GROUP_FALLBACK)
            has_hate = normalize_label(has_hate_raw, 'has_hate',
                                       has_hate_labels_set, DEFAULT_HAS_HATE_FALLBACK)

            formatted_quads.append(
                f"{target} | {argument} | {targeted_group} | {has_hate} [END]"
            )
        else:
            # 对于无法解析的四元组，严格模式下不添加，或者可以添加一个默认/警告项
            pass  # 严格模式：只添加符合格式的，不符合的忽略
    # 如果没有任何四元组符合格式，则输出一个默认值，避免空行
    return "[SEP]".join(formatted_quads) if formatted_quads else "无 | 无 | 未知 | 未知 [END]"


# --- 2. 加载模型和分词器 ---
print("加载基础模型和分词器...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

# 分词器从基础模型加载
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载微调后的 LoRA 适配器
print(f"加载 LoRA 适配器 {LORA_ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model_base, LORA_ADAPTER_DIR)
model.eval()  # 设置为评估模式

print("模型加载完成。")

# --- 3. 读取测试数据 ---
print(f"读取测试文件 {TEST_FILE}...")
test_data = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"总共读取 {len(test_data)} 条测试数据。将处理全部数据。")

# --- 4. 进行预测并输出 ---
print(f"开始预测并将结果写入 {OUTPUT_TXT_FILE}...")
with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as outfile:
    # 循环遍历所有测试数据
    for i, item in enumerate(test_data):
        content = item['content']

        # 构建 Qwen-Chat 的指令格式，包含训练时使用的标签限制提示
        # 这里使用手动定义的 COLLECTED_TARGETED_GROUPS 和 COLLECTED_HAS_HATE_LABELS
        instruction = (
            f"请根据以下社交媒体文本，识别出其中包含的细粒度仇恨言论四元组，包括评论对象、论点、目标群体和是否具有仇恨。"
            f"如果评论对象不存在，请标记为“无”。如果存在多个四元组，请使用“[SEP]”进行分隔。"
            f"每个四元组的格式为：评论对象 | 论点 | 目标群体 | 是否具有仇恨 [END]。\n"
            f"注意：'目标群体'字段必须是以下标签之一：{sorted(list(COLLECTED_TARGETED_GROUPS))}。\n"
            f"注意：'是否具有仇恨'字段必须是以下标签之一：{sorted(list(COLLECTED_HAS_HATE_LABELS))}。\n\n"
            f"文本：{content}"
        )
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,  # 如果格式依然不稳定，可以尝试降低到 0.5-0.6
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 解析并规范化模型输出，以便格式化
        formatted_output_line = parse_and_format_generated_output(generated_text, COLLECTED_TARGETED_GROUPS,
                                                                  COLLECTED_HAS_HATE_LABELS)

        outfile.write(formatted_output_line + "\n")

        # 每40条打印一次进度
        if (i + 1) % 40 == 0:
            print(f"已处理 {i + 1} 条数据...")

    print(f"已处理 {len(test_data)} 条数据。")

print(f"预测完成，结果已保存到 {OUTPUT_TXT_FILE}")