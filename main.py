import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import os
from collections import Counter
import re

# --- 0. 配置参数 ---
# 修改为 Qwen3-1.7B
INITIAL_MODEL_NAME_FOR_CONFIG = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./qwen_finetuned_hate_speech_v4_qwen3"  # 修改输出目录以区分版本和模型
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_lora_adapters")
TRAIN_FILE = 'train.json'
MAX_SEQ_LENGTH = 512
BATCH_SIZE_PER_DEVICE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
EVAL_STEPS = 200
LOGGING_STEPS = 50

# 确保CUDA可用
if not torch.cuda.is_available():
    raise SystemError("CUDA 不可用。请确保您有兼容的 GPU 并已安装 CUDA。")
print(f"正在使用设备: {torch.cuda.current_device()}")

# --- 1. 数据处理 ---

# 默认回退值，当标签无法规范化时使用
DEFAULT_TARGET_GROUP_FALLBACK = "未知群体"
DEFAULT_HAS_HATE_FALLBACK = "未知"


def _parse_raw_output_to_temp_quads(output_str):
    """
    临时解析原始 output 字符串，提取所有四元组的原始部分。
    """
    quadruple_strs = output_str.split('[SEP]')
    temp_quads = []
    for q_str in quadruple_strs:
        q_str = q_str.replace('[END]', '').strip()
        parts = [p.strip() for p in q_str.split('|')]
        if len(parts) == 4:
            temp_quads.append({
                'original_target': parts[0],
                'original_argument': parts[1],
                'original_targeted_group': parts[2],
                'original_hateful_label': parts[3]
            })
    return temp_quads


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
                if "是" in allowed_labels_set and (
                        lower_label == '是' or lower_label in {"yes", "true", "有", "hate", "hateful"}):
                    return "是"
                if "否" in allowed_labels_set and (
                        lower_label == '否' or lower_label in {"no", "false", "not hate", "无", "中性", "neutral"}):
                    return "否"
            if label_str.lower() == 'null' and field_name == 'target':
                return "无"

            return fallback_value
    else:
        if label_str.lower() == 'null':
            return "无"
        return label_str


def load_and_prepare_data(json_file_path):
    """
    加载并准备数据为 Qwen-Chat 模型所需的格式。
    动态收集 '目标群体' 和 '是否具有仇恨' 的标签。
    '评论对象' 和 '论点' 视为自由文本，不进行标签收集。
    """
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_targeted_groups_found = set()
    raw_has_hate_labels_found = set()

    for item in data:
        temp_quads = _parse_raw_output_to_temp_quads(item['output'])
        for q in temp_quads:
            raw_targeted_groups_found.add(q['original_targeted_group'].strip())
            raw_has_hate_labels_found.add(q['original_hateful_label'].strip())

    global_allowed_targeted_groups = sorted(list(raw_targeted_groups_found))
    global_allowed_has_hate_labels = sorted(list(raw_has_hate_labels_found))

    print("\n--- 从数据集中动态收集到的所有唯一 '目标群体' 标签 ---")
    for label in global_allowed_targeted_groups: print(f"- {label}")
    print(f"总共发现 {len(global_allowed_targeted_groups)} 种唯一 '目标群体' 标签。")

    print("\n--- 从数据集中动态收集到的所有唯一 '是否具有仇恨' 标签 ---")
    for label in global_allowed_has_hate_labels: print(f"- {label}")
    print(f"总共发现 {len(global_allowed_has_hate_labels)} 种唯一 '是否具有仇恨' 标签。")
    print("---------------------------------------------------\n")

    aggregated_data = {}
    for item in data:
        content = item['content']
        quadruple_strs = item['output'].split('[SEP]')

        formatted_quads = []
        for q_str in quadruple_strs:
            q_str = q_str.replace('[END]', '').strip()
            parts = [p.strip() for p in q_str.split('|')]

            if len(parts) == 4:
                original_target, original_argument, original_targeted_group, original_hateful_label = parts

                target = normalize_label(original_target, 'target')
                argument = normalize_label(original_argument, 'argument')

                targeted_group = normalize_label(original_targeted_group, 'targeted_group',
                                                 set(global_allowed_targeted_groups), DEFAULT_TARGET_GROUP_FALLBACK)
                has_hate = normalize_label(original_hateful_label, 'has_hate',
                                           set(global_allowed_has_hate_labels), DEFAULT_HAS_HATE_FALLBACK)

                formatted_quads.append(
                    f"{target} | {argument} | {targeted_group} | {has_hate} [END]"
                )
            else:
                print(f"警告: 发现无法解析的原始四元组字符串: '{q_str}'. 跳过此项。")

        full_output = "[SEP]".join(formatted_quads)

        instruction = (
            f"请根据以下社交媒体文本，识别出其中包含的细粒度仇恨言论四元组，包括评论对象、论点、目标群体和是否具有仇恨。"
            f"如果评论对象不存在，请标记为“无”。如果存在多个四元组，请使用“[SEP]”进行分隔。"
            f"每个四元组的格式为：评论对象 | 论点 | 目标群体 | 是否具有仇恨 [END]。\n"
            f"注意：'目标群体'字段必须是以下标签之一：{global_allowed_targeted_groups}。\n"
            f"注意：'是否具有仇恨'字段必须是以下标签之一：{global_allowed_has_hate_labels}。\n\n"
            f"文本：{content}"
        )
        training_text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{full_output}<|im_end|>"
        )
        aggregated_data[content] = training_text

    processed_list = [{"text": text} for text in aggregated_data.values()]

    df_raw = pd.DataFrame([
        {'id': item['id'], 'content': item['content'], 'original_output': item['output']}
        for item in data
    ])

    return (processed_list, df_raw,
            set(global_allowed_targeted_groups),
            set(global_allowed_has_hate_labels))


print("开始数据处理...")
(training_data_list, df_original_eval,
 collected_targeted_groups, collected_has_hate_labels) = load_and_prepare_data(TRAIN_FILE)
print(f"处理后用于训练的总条数: {len(training_data_list)}")

unique_contents = df_original_eval['content'].unique().tolist()
train_contents, val_contents = train_test_split(unique_contents, test_size=0.1, random_state=42)

train_dataset_raw = [item for item in training_data_list if any(content in item['text'] for content in train_contents)]
val_dataset_raw = [item for item in training_data_list if any(content in item['text'] for content in val_contents)]

val_df_for_eval = df_original_eval[df_original_eval['content'].isin(val_contents)].reset_index(drop=True)

print(f"训练集大小: {len(train_dataset_raw)}")
print(f"验证集大小: {len(val_dataset_raw)}")
print("训练数据示例（第一条）：")
print(train_dataset_raw[0]['text'])


# --- Custom Dataset Class ---
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone()
        }


# --- 2. 模型微调 ---
print("加载模型和分词器...")
# 确保这里使用 Qwen3-1.7B
model_name = r"C:\Users\23844\PycharmProjects\PythonProject\Qwen3-1___7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = CustomDataset(train_dataset_raw, tokenizer, MAX_SEQ_LENGTH)
val_dataset = CustomDataset(val_dataset_raw, tokenizer, MAX_SEQ_LENGTH)

# --- 3. 定义训练参数 ---
print("\n定义训练参数...")
training_args = TrainingArguments(
    output_dir="./results_qwen3",  # 修改输出目录
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
    warmup_steps=500,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs_qwen3",  # 修改日志目录
    logging_steps=LOGGING_STEPS,
    fp16=True,
    report_to="none",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("\n开始训练...")
trainer.train()

# --- 4. 保存微调后的模型 ---
os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)
print(f"\n保存 LoRA 适配器到：{LORA_ADAPTER_DIR}")
trainer.model.save_pretrained(LORA_ADAPTER_DIR)
tokenizer.save_pretrained(LORA_ADAPTER_DIR)

print("训练完成。")


# --- 5. 评估 ---
def parse_generated_output(generated_text, allowed_labels_dict):
    """
    尝试从模型生成的文本中解析出四元组列表，并对标签进行规范化。
    """
    quadruples = []
    segments = generated_text.split('[SEP]')
    for seg in segments:
        seg = seg.replace('[END]', '').strip()
        parts = [p.strip() for p in seg.split('|')]

        if len(parts) == 4:
            target_raw, argument_raw, targeted_group_raw, has_hate_raw = parts

            target = normalize_label(target_raw, 'target')
            argument = normalize_label(argument_raw, 'argument')
            targeted_group = normalize_label(targeted_group_raw, 'targeted_group',
                                             allowed_labels_dict['targeted_groups'], DEFAULT_TARGET_GROUP_FALLBACK)
            has_hate = normalize_label(has_hate_raw, 'has_hate',
                                       allowed_labels_dict['has_hate_labels'], DEFAULT_HAS_HATE_FALLBACK)

            quadruples.append({
                'target': target,
                'argument': argument,
                'targeted_group': targeted_group,
                'has_hate': has_hate
            })
    return quadruples


def evaluate_quadruples(predictions_df, tokenizer, current_model_name, lora_adapter_path, bnb_config,
                        allowed_labels_dict_for_eval):
    """
    对模型在验证集上的性能进行评估，比较预测的四元组和真实的四元组。
    """
    print("\n开始评估...")
    print(f"加载基础模型 {current_model_name}...")

    model_base = AutoModelForCausalLM.from_pretrained(
        current_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"加载 LoRA 适配器 {lora_adapter_path}...")
    model_peft = PeftModel.from_pretrained(model_base, lora_adapter_path)
    model_peft.eval()

    all_true_quadruples = []
    all_pred_quadruples = []

    print("\n评估时依据的标签集合:")
    print(f"  目标群体: {sorted(list(allowed_labels_dict_for_eval['targeted_groups']))}")
    print(f"  是否具有仇恨: {sorted(list(allowed_labels_dict_for_eval['has_hate_labels']))}")

    for index, row in predictions_df.iterrows():
        content = row['content']
        original_output_str = row['original_output']

        def parse_original_for_eval(raw_output_str, allowed_labels_dict_eval):
            quads = []
            for q_str in raw_output_str.split('[SEP]'):
                q_str = q_str.replace('[END]', '').strip()
                parts = [p.strip() for p in q_str.split('|')]
                if len(parts) == 4:
                    original_target, original_argument, original_targeted_group, original_hateful_label = parts

                    target = normalize_label(original_target, 'target')
                    argument = normalize_label(original_argument, 'argument')

                    targeted_group = normalize_label(original_targeted_group, 'targeted_group',
                                                     allowed_labels_dict_eval['targeted_groups'],
                                                     DEFAULT_TARGET_GROUP_FALLBACK)
                    has_hate = normalize_label(original_hateful_label, 'has_hate',
                                               allowed_labels_dict_eval['has_hate_labels'], DEFAULT_HAS_HATE_FALLBACK)
                    quads.append({
                        'target': target,
                        'argument': argument,
                        'targeted_group': targeted_group,
                        'has_hate': has_hate
                    })
            return quads

        true_quadruples_parsed = parse_original_for_eval(original_output_str, allowed_labels_dict_for_eval)

        instruction = (
            f"请根据以下社交媒体文本，识别出其中包含的细粒度仇恨言论四元组，包括评论对象、论点、目标群体和是否具有仇恨。"
            f"如果评论对象不存在，请标记为“无”。如果存在多个四元组，请使用“[SEP]”进行分隔。"
            f"每个四元组的格式为：评论对象 | 论点 | 目标群体 | 是否具有仇恨 [END]。\n"
            f"注意：'目标群体'字段必须是以下标签之一：{sorted(list(allowed_labels_dict_for_eval['targeted_groups']))}。\n"
            f"注意：'是否具有仇恨'字段必须是以下标签之一：{sorted(list(allowed_labels_dict_for_eval['has_hate_labels']))}。\n\n"
            f"文本：{content}"
        )
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LENGTH).to(
            model_peft.device)

        with torch.no_grad():
            outputs = model_peft.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predicted_quadruples = parse_generated_output(generated_text, allowed_labels_dict_for_eval)

        all_true_quadruples.append(true_quadruples_parsed)
        all_pred_quadruples.append(predicted_quadruples)

    total_true_quads = sum(len(q_list) for q_list in all_true_quadruples)
    total_pred_quads = sum(len(q_list) for q_list in all_pred_quadruples)

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for i in range(len(all_true_quadruples)):
        true_quads = all_true_quadruples[i]
        pred_quads = all_pred_quadruples[i]

        true_counts = Counter(tuple(sorted(q.items())) for q in true_quads)
        pred_counts = Counter(tuple(sorted(q.items())) for q in pred_quads)

        temp_tp = 0
        for q_tuple in true_counts:
            if q_tuple in pred_counts:
                match_count = min(true_counts[q_tuple], pred_counts[q_tuple])
                temp_tp += match_count
                pred_counts[q_tuple] -= match_count
                true_counts[q_tuple] -= match_count

        overall_tp += temp_tp
        overall_fp += sum(pred_counts.values())
        overall_fn += sum(true_counts.values())

    print(f"\n--- 评估结果 ---")
    print(f"验证集样本数: {len(predictions_df)}")
    print(f"总真实四元组数: {total_true_quads}")
    print(f"总预测四元组数: {total_pred_quads}")

    exact_match_accuracy = overall_tp / total_true_quads if total_true_quads > 0 else 0
    print(f"精确匹配四元组数 (TP): {overall_tp}")
    print(f"精确匹配率 (Exact Match Accuracy/Recall): {exact_match_accuracy:.4f}")

    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n整体四元组匹配 (Precision/Recall/F1):")
    print(f"  True Positives (TP): {overall_tp}")
    print(f"  False Positives (FP): {overall_fp}")
    print(f"  False Negatives (FN): {overall_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

    print("\n--- 示例预测（前5个） ---")
    for i in range(min(5, len(predictions_df))):
        content = predictions_df.iloc[i]['content']
        original_output = predictions_df.iloc[i]['original_output']

        instruction = (
            f"请根据以下社交媒体文本，识别出其中包含的细粒度仇恨言论四元组，包括评论对象、论点、目标群体和是否具有仇恨。"
            f"如果评论对象不存在，请标记为“无”。如果存在多个四元组，请使用“[SEP]”进行分隔。"
            f"每个四元组的格式为：评论对象 | 论点 | 目标群体 | 是否具有仇恨 [END]。\n"
            f"注意：'目标群体'字段必须是以下标签之一：{sorted(list(allowed_labels_dict_for_eval['targeted_groups']))}。\n"
            f"注意：'是否具有仇恨'字段必须是以下标签之一：{sorted(list(allowed_labels_dict_for_eval['has_hate_labels']))}。\n\n"
            f"文本：{content}"
        )
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LENGTH).to(
            model_peft.device)
        with torch.no_grad():
            outputs = model_peft.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        true_quads_parsed_for_display = parse_original_for_eval(original_output, allowed_labels_dict_for_eval)
        display_true_output_quads = [
            f"{q['target']} | {q['argument']} | {q['targeted_group']} | {q['has_hate']} [END]"
            for q in true_quads_parsed_for_display
        ]
        display_original_output = "[SEP]".join(display_true_output_quads)

        print(f"\n文本：{content}")
        print(f"真实输出（规范化后）：{display_original_output}")
        print(f"模型预测：{generated_text}")
        print("-" * 30)


all_allowed_labels_for_eval = {
    'targeted_groups': collected_targeted_groups,
    'has_hate_labels': collected_has_hate_labels
}

evaluate_quadruples(val_df_for_eval, tokenizer, model_name, LORA_ADAPTER_DIR, bnb_config, all_allowed_labels_for_eval)