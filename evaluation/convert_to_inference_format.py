#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将原始episode格式转换为推理评估所需的格式

原始格式:
{
  "episode_id": "...",
  "instruction": "...",
  "sub_instructions": [...],
  "acts_origin": [...],
  "acts_convert": [...],
  "img": ["/path/to/image1.png", ...],
  "client_number": "..."
}

推理格式:
{
  "images": ["/path/to/image1.png", ...],
  "query": "<image>\n<image>\n指令内容",
  "response": "动作1\n动作2\n..."  # 作为label
}
"""

import argparse
import json
import os
from typing import Any, Dict, List


def stringify_list(items: Any) -> str:
    """将列表转换为字符串，每行一个元素"""
    if isinstance(items, list):
        parts: List[str] = []
        for x in items:
            if isinstance(x, (dict, list)):
                parts.append(json.dumps(x, ensure_ascii=False))
            else:
                parts.append(str(x))
        return '\n'.join(parts)
    return str(items) if items else ''


def build_query(instruction: str, image_count: int, image_tag: str = '<image>') -> str:
    """构建query，包含图片标签和指令"""
    tags = '\n'.join([image_tag] * image_count) if image_count > 0 else ''
    if tags:
        return f"{tags}\n{instruction}".strip()
    return instruction or ''


def convert_record(d: Dict[str, Any], *, image_root: str = None, image_prefix_src: str = None, 
                   image_prefix_dst: str = None, drop_if_missing: bool = False) -> Dict[str, Any]:
    """
    转换单条记录
    
    Args:
        d: 原始数据记录
        image_root: 图片路径根目录（用于处理相对路径）
        image_prefix_src: 要替换的图片路径前缀（源前缀）
        image_prefix_dst: 替换后的图片路径前缀（目标前缀）
        drop_if_missing: 如果图片不存在是否丢弃该样本
    
    Returns:
        转换后的记录，如果drop_if_missing=True且图片不存在则返回None
    """
    # 获取图片路径列表
    raw_images = d.get('img') or d.get('imgs') or d.get('images') or []
    if isinstance(raw_images, str):
        images = [raw_images]
    elif isinstance(raw_images, list):
        images = raw_images
    else:
        images = []
    
    # 处理图片路径
    processed_images = []
    for img_path in images:
        if isinstance(img_path, str):
            # 路径前缀替换（优先处理）
            if image_prefix_src and image_prefix_dst:
                # 确保前缀以/结尾，以便正确替换
                src_prefix = image_prefix_src.rstrip('/') + '/'
                dst_prefix = image_prefix_dst.rstrip('/') + '/'
                if img_path.startswith(src_prefix):
                    img_path = dst_prefix + img_path[len(src_prefix):]
                elif img_path.startswith(image_prefix_src):
                    # 如果原始路径不以/结尾，也尝试匹配
                    img_path = image_prefix_dst + img_path[len(image_prefix_src):]
            # 如果是相对路径且提供了image_root，则拼接（仅在未进行前缀替换时）
            elif image_root and not os.path.isabs(img_path):
                img_path = os.path.join(image_root, img_path)
            processed_images.append(img_path)
    
    # 检查图片是否存在
    existing_images = [p for p in processed_images if os.path.exists(p)] if processed_images else []
    
    if drop_if_missing and not existing_images:
        return None
    
    # 使用存在的图片，如果drop_if_missing=False则使用原始路径
    final_images = existing_images if existing_images else processed_images
    
    # 获取指令
    instruction = d.get('instruction') or ''
    
    # 构建query（包含图片标签和指令）
    query = build_query(instruction, len(final_images))
    
    # 获取response（作为label）
    # 优先使用acts_convert，其次acts_origin，最后其他字段
    if d.get('acts_convert'):
        response = stringify_list(d['acts_convert'])
    elif d.get('acts_origin'):
        # 将acts_origin转换为文本
        response = stringify_list(d['acts_origin'])
    elif d.get('response'):
        response = d['response']
    elif d.get('label'):
        response = d['label']
    else:
        response = ''
    
    # 构建输出记录
    result = {
        'images': final_images,
        'query': query,
        'response': response  # 推理时会将其作为label
    }
    
    # 可选：保留episode_id等元信息
    if 'episode_id' in d:
        result['episode_id'] = d['episode_id']
    
    return result


def convert_file(input_path: str, output_path: str, *, image_root: str = None, 
                 image_prefix_src: str = None, image_prefix_dst: str = None, 
                 drop_if_missing: bool = False) -> tuple:
    """
    转换整个文件
    
    Returns:
        (输入样本数, 输出样本数)
    """
    # 读取输入文件
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效的JSON行: {e}")
                continue
    
    # 转换记录
    converted = []
    dropped = 0
    for record in records:
        converted_record = convert_record(
            record, 
            image_root=image_root, 
            image_prefix_src=image_prefix_src,
            image_prefix_dst=image_prefix_dst,
            drop_if_missing=drop_if_missing
        )
        if converted_record is not None:
            converted.append(converted_record)
        else:
            dropped += 1
    
    # 保存输出文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    if dropped > 0:
        print(f'警告: 丢弃了 {dropped} 个样本（图片不存在）')
    
    return len(records), len(converted)


def main():
    parser = argparse.ArgumentParser(
        description='将原始episode格式转换为推理评估格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换
  python convert_to_inference_format.py --input data.jsonl --output Val_100.jsonl
  
  # 处理相对路径
  python convert_to_inference_format.py --input data.jsonl --output Val_100.jsonl --image_root /path/to/images
  
  # 替换路径前缀（将绝对路径转换为相对路径）
  python convert_to_inference_format.py --input data.jsonl --output Val_100.jsonl \\
      --image_prefix_src "/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/unpack-androidcontrol/" \\
      --image_prefix_dst "./android_control_unpack/"
  
  # 丢弃图片不存在的样本
  python convert_to_inference_format.py --input data.jsonl --output Val_100.jsonl --drop_if_missing
        """
    )
    parser.add_argument('--input', '-i', required=True, help='输入文件路径 (.jsonl)')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径 (.jsonl)')
    parser.add_argument('--image_root', default=None, help='图片路径根目录（用于处理相对路径）')
    parser.add_argument('--image_prefix_src', default=None, help='要替换的图片路径前缀（源前缀）')
    parser.add_argument('--image_prefix_dst', default=None, help='替换后的图片路径前缀（目标前缀）')
    parser.add_argument('--drop_if_missing', action='store_true', help='如果图片不存在则丢弃该样本')
    
    args = parser.parse_args()
    
    # 检查参数
    if (args.image_prefix_src is None) != (args.image_prefix_dst is None):
        print("错误: --image_prefix_src 和 --image_prefix_dst 必须同时指定或都不指定")
        return
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print(f"正在转换: {args.input} -> {args.output}")
    if args.image_prefix_src and args.image_prefix_dst:
        print(f"路径前缀替换: {args.image_prefix_src} -> {args.image_prefix_dst}")
    
    n_in, n_out = convert_file(
        args.input,
        args.output,
        image_root=args.image_root,
        image_prefix_src=args.image_prefix_src,
        image_prefix_dst=args.image_prefix_dst,
        drop_if_missing=args.drop_if_missing
    )
    
    print(f"✅ 转换完成: {n_in} -> {n_out} 个样本")
    print(f"   输出文件: {args.output}")


if __name__ == '__main__':
    main()

