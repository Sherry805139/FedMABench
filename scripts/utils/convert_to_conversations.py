import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(file_path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def read_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError('JSON file must contain a top-level list of objects.')


def stringify_list(items: Any) -> str:
    if isinstance(items, list):
        parts: List[str] = []
        for x in items:
            if isinstance(x, (dict, list)):
                parts.append(json.dumps(x, ensure_ascii=False))
            else:
                parts.append(str(x))
        return '\n'.join(parts)
    return str(items)


def acts_origin_to_text(acts_origin: Any) -> str:
    if not acts_origin:
        return ''
    lines: List[str] = []
    for a in acts_origin:
        try:
            obj = json.loads(a) if isinstance(a, str) else a
            if isinstance(obj, dict):
                action_type = obj.get('action_type', '')
                if action_type == 'click':
                    lines.append(f"Click at (x={obj.get('x')}, y={obj.get('y')})")
                elif action_type == 'input_text':
                    lines.append(f"Type text: {obj.get('text', '')}")
                elif action_type == 'open_app':
                    lines.append(f"Open App: {obj.get('app_name', '')}")
                elif action_type == 'status':
                    lines.append(f"status: {obj.get('goal_status')}")
                else:
                    lines.append(json.dumps(obj, ensure_ascii=False))
            else:
                lines.append(str(obj))
        except Exception:
            lines.append(str(a))
    return '\n'.join(lines)


def build_user_value(instruction: str, image_count: int, image_tag: str = '<image>') -> str:
    tags = '\n'.join([image_tag] * image_count) if image_count > 0 else ''
    if tags:
        return f"{tags}\n{instruction}".strip()
    return instruction or ''


def _remap_images(images: List[str], image_prefix_src: str = None, image_prefix_dst: str = None,
                  image_root: str = None) -> List[str]:
    remapped: List[str] = []
    for p in images:
        new_p = p
        if image_prefix_src and image_prefix_dst and isinstance(new_p, str) and new_p.startswith(image_prefix_src):
            new_p = image_prefix_dst + new_p[len(image_prefix_src):]
        if image_root and isinstance(new_p, str) and not os.path.isabs(new_p):
            new_p = os.path.join(image_root, new_p)
        remapped.append(new_p)
    return remapped


def convert_record(d: Dict[str, Any], *, image_prefix_src: str = None, image_prefix_dst: str = None,
                   image_root: str = None, drop_if_missing: bool = False) -> Dict[str, Any]:
    # images
    raw_images = d.get('imgs') or d.get('images') or []
    images: List[str] = raw_images if isinstance(raw_images, list) else [raw_images]
    images = _remap_images(images, image_prefix_src, image_prefix_dst, image_root)
    existing_images = [p for p in images if isinstance(p, str) and os.path.exists(p)] if images else []

    # instruction + images => user
    instruction: str = d.get('instruction') or ''
    # Prefer counting existing images when deciding how many <image> tags to place
    user_value = build_user_value(instruction, len(existing_images) if existing_images else len(images))

    # response: prefer acts_convert, else acts_origin summary
    if d.get('acts_convert'):
        response_value = stringify_list(d['acts_convert'])
    elif d.get('acts_origin'):
        response_value = acts_origin_to_text(d['acts_origin'])
    else:
        response_value = d.get('response') or d.get('output') or ''

    conversations = [
        {'from': 'user', 'value': user_value},
        {'from': 'assistant', 'value': response_value},
    ]

    out: Dict[str, Any] = {'conversations': conversations}
    # Keep only files that actually exist, unless user wants to keep raw paths
    if existing_images:
        out['images'] = existing_images
    elif images and drop_if_missing:
        # Mark as invalid by returning empty dict; caller will filter
        return {}

    # keep useful meta fields
    for meta_key in ('episode_id', 'client_number'):
        if meta_key in d:
            out[meta_key] = d[meta_key]

    return out


def convert_file(input_path: str, output_path: str, *, image_prefix_src: str = None, image_prefix_dst: str = None,
                 image_root: str = None, drop_if_missing: bool = False) -> Tuple[int, int]:
    if input_path.endswith('.jsonl'):
        items = read_jsonl(input_path)
    elif input_path.endswith('.json'):
        items = read_json(input_path)
    else:
        raise ValueError('Input must be .json or .jsonl')

    converted: List[Dict[str, Any]] = []
    dropped = 0
    for x in items:
        y = convert_record(
            x,
            image_prefix_src=image_prefix_src,
            image_prefix_dst=image_prefix_dst,
            image_root=image_root,
            drop_if_missing=drop_if_missing,
        )
        if y:
            converted.append(y)
        else:
            dropped += 1
    write_jsonl(output_path, converted)
    if dropped:
        print(f'Dropped {dropped} samples due to missing images.')
    return len(items), len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert dataset to conversations format for Qwen2-VL.')
    parser.add_argument('--input', required=True, help='Input file (.json/.jsonl) or directory')
    parser.add_argument('--output', required=True, help='Output file (.jsonl) or directory when input is a dir')
    parser.add_argument('--image_prefix_src', default=None, help='Old absolute prefix to replace in image paths')
    parser.add_argument('--image_prefix_dst', default=None, help='New absolute prefix to replace with')
    parser.add_argument('--image_root', default=None, help='Join non-absolute image paths with this root')
    parser.add_argument('--drop_if_missing', action='store_true', help='Drop samples if images not found locally')
    args = parser.parse_args()

    src = args.input
    dst = args.output

    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        total_in, total_out = 0, 0
        for root, _, files in os.walk(src):
            for name in files:
                if not (name.endswith('.json') or name.endswith('.jsonl')):
                    continue
                in_path = os.path.join(root, name)
                rel = os.path.relpath(in_path, src)
                out_name = os.path.splitext(rel)[0] + '.jsonl'
                out_path = os.path.join(dst, out_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                n_in, n_out = convert_file(
                    in_path,
                    out_path,
                    image_prefix_src=args.image_prefix_src,
                    image_prefix_dst=args.image_prefix_dst,
                    image_root=args.image_root,
                    drop_if_missing=args.drop_if_missing,
                )
                total_in += n_in
                total_out += n_out
        print(f'Converted {total_in} -> {total_out} items across files.')
    else:
        if os.path.isdir(dst):
            base = os.path.basename(src)
            out_name = os.path.splitext(base)[0] + '.jsonl'
            out_path = os.path.join(dst, out_name)
        else:
            out_path = dst
        n_in, n_out = convert_file(
            src,
            out_path,
            image_prefix_src=args.image_prefix_src,
            image_prefix_dst=args.image_prefix_dst,
            image_root=args.image_root,
            drop_if_missing=args.drop_if_missing,
        )
        print(f'Converted {n_in} -> {n_out} items to {out_path}')


if __name__ == '__main__':
    main()


