import os
from tqdm import tqdm
import json
from eval_gpt import calculate_tfidf
import argparse


def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def judge_step(a, b):
    if calculate_tfidf(a, b) > 0.6:
        return 1
    else:
        return 0


def calculate_step_accuracy(data):
    """
    计算步骤级准确率：将 response 和 label 按行分割，逐行比较每个动作步骤。
    返回每个步骤的准确性列表 [1, 0, 1, ...]。
    """
    step_accuracies = []
    for item in data:
        # 将 response 和 label 按行分割成步骤列表
        response_steps = [step.strip() for step in item['response'].split('\n') if step.strip()]
        label_steps = [step.strip() for step in item['label'].split('\n') if step.strip()]
        
        # 逐步骤比较，取较短的序列长度
        min_len = min(len(response_steps), len(label_steps))
        if min_len == 0:
            continue
            
        # 对每个步骤进行比较
        for i in range(min_len):
            # 跳过 "Check status: successful" 这类状态检查步骤
            if 'Check status' in response_steps[i] or 'Check status' in label_steps[i]:
                continue
            step_accuracies.append(judge_step(label_steps[i], response_steps[i]))
    
    return step_accuracies


def calculate_episode_accuracy(data, step_accuracies=None):
    """
    按照 query 对数据进行分组，然后计算每个 episode 的准确率。
    如果一个 episode 中所有行的 label 和 response 都相同，准确率为 1，否则为 0。
    """


    episode_accuracies = []
    episodes = {}
    import re


    # 按照 query 对数据进行分组
    for item in data:
        query = item['query']
        
        # 尝试提取 User Instruction（如果格式是 ### User Instruction ###\n...\n###）
        match = re.search(r'### User Instruction ###\n(.*?)\n###', query, re.DOTALL)
        if match:
            query = match.group(1).strip()  # 提取 User Instruction 部分并去除首尾空格
        else:
            # 如果没有找到标准格式，尝试提取 <image> 标签后的指令部分
            # 格式通常是: <image>\n<image>\n...\n指令内容
            lines = query.split('\n')
            # 找到第一个不是 <image> 的行，作为指令开始
            instruction_lines = []
            found_instruction = False
            for line in lines:
                line_stripped = line.strip()
                if line_stripped == '<image>' or line_stripped == '':
                    continue
                # 找到第一个非空且不是 <image> 的行，后面的都是指令
                found_instruction = True
                instruction_lines.append(line)
            
            if instruction_lines:
                # 合并所有指令行
                query = '\n'.join(instruction_lines).strip()
            # 如果还是找不到，就使用原始 query（去掉开头的 <image> 标签）
            if not query or query == item['query']:
                # 移除所有 <image> 行，保留其他内容
                query = '\n'.join([line for line in lines if line.strip() != '<image>']).strip()
        
        if query not in episodes:
            episodes[query] = []
        episodes[query].append(item)

    # 对每个 episode 检查所有行是否相关
    for query, episode_items in episodes.items():
        # 如果 episode 中所有行的准确性都为 1，则认为该 episode 的准确率为 1
        if all(judge_step(item['label'], item['response']) for item in episode_items):
            episode_accuracies.append(1)
        else:
            episode_accuracies.append(0)

    return episode_accuracies


def test_main(data_path):
    # 读取数据
    # data_path = r'/ailab/user/wangwenhao/FedMobile/output/qwen2-vl-7b-instruct/v7-20241219-094924/global_lora_49/infer_result/20241220-001128.jsonl'
    data = read_jsonl(data_path)
    
    # 获取数据所在目录和文件名
    output_dir = os.path.dirname(data_path)
    base_filename = os.path.splitext(os.path.basename(data_path))[0]  # 获取不带后缀的文件名
    output_file = os.path.join(output_dir, base_filename + '.log')    # 构建 .log 文件路径

    # 打开文件保存输出

    # 计算每一行的准确率
    step_accuracies = calculate_step_accuracy(data)
    step_accuracy = sum(step_accuracies) / len(step_accuracies)
    print(f"Step-level accuracy: {step_accuracy * 100:.2f}%")

    # 计算按 query 分组的 episode 准确率
    episode_accuracies = calculate_episode_accuracy(data)
    episode_accuracy = sum(episode_accuracies) / len(episode_accuracies)
    print(f"len: {len(episode_accuracies)}")
    print(f"Episode-level accuracy: {episode_accuracy * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    # parser.add_argument("--save_failed_generation", action='store_true', default=False)

    args = parser.parse_args()

    test_main(args.data_path)
