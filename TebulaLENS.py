# -*- coding: utf-8 -*-

import os
import sys
import base64
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import configparser
from openai import OpenAI

# --- 1. 配置加载 (适配DashScope API) ---

def setup_configuration(config_path: Path) -> dict:
    """加载或创建配置文件，并返回配置参数字典。"""
    if not config_path.exists():
        print(f"[信息] 配置文件 '{config_path.name}' 不存在，正在创建默认配置...")
        config = configparser.ConfigParser()
        config['General'] = {'serialization_enabled': 'True'}
        config['AliyunDashScope'] = {
            'api_key': '',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'vision_model': 'qwen-vl-max',
            'text_model': 'qwen3-30b-a3b-thinking-2507'
        }
        config['Paths'] = {
            'pdf_source_dir': 'pdf', 'image_cache_dir': 'PdfImages', 'output_dir': 'txt',
            'serialization_output_dir': 'NL_serialize', 'combination_output_dir': 'Combined_output'
        }
        config['LLM_Settings'] = {
            'stream_token_limit': '12288', 'loop_check_deque_size': '300', 'loop_detection_threshold': '100'
        }
        with config_path.open('w', encoding='utf-8') as f:
            f.write("# PDF表格提取与序列化工具配置文件\n# 详细说明请参考脚本或文档\n\n")
            config.write(f)
        print(f"[完成] 已创建默认配置文件 '{config_path.name}'。请填入API Key后重新运行。")
        sys.exit()

    print(f"[信息] 正在从 '{config_path.name}' 加载配置...")
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    settings = {}
    script_dir = config_path.parent
    
    settings['serialization_enabled'] = config.getboolean('General', 'serialization_enabled', fallback=False)
    
    # 读取DashScope配置
    api_cfg = config['AliyunDashScope']
    # 优先从环境变量读取API Key，其次从配置文件读取
    settings['api_key'] = os.getenv('DASHSCOPE_API_KEY') or api_cfg.get('api_key')
    settings['base_url'] = api_cfg.get('base_url')
    settings['vision_model'] = api_cfg.get('vision_model')
    settings['text_model'] = api_cfg.get('text_model')

    if not settings['api_key']:
        raise ValueError("API Key未找到。请设置 DASHSCOPE_API_KEY 环境变量或在config.ini中填写。")

    paths_cfg = config['Paths']
    for key, value in paths_cfg.items():
        settings[key] = script_dir / value

    print("[信息] 配置加载成功。")
    return settings

# --- 2. 辅助函数 ---

def load_prompt(file_path: Path) -> str:
    if not file_path.is_file(): raise FileNotFoundError(f"指令文件未找到: {file_path}")
    return file_path.read_text(encoding='utf-8')

def encode_image_base64(image_path: Path) -> str:
    """将图片文件转换为Base64编码的字符串。"""
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    image_dir = output_dir / pdf_path.stem
    if image_dir.exists():
        images = sorted(list(image_dir.glob("page_*.png")), key=lambda p: int(p.stem.split('_')[1]))
        if images: print(f"  -> 使用已缓存的图片于: {image_dir}"); return images
    print(" > 未发现已缓存的渲染结果，正在将pdf渲染为图片(对于大型pdf，这一步耗时较长)...")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(8, 8)) # 高分辨率
        image_path = image_dir / f"page_{page_num + 1}.png"
        pix.save(image_path, "png")
        image_paths.append(image_path)
    doc.close()
    print(f"  -> 图片转换完成: {image_dir}")
    return image_paths

# --- 3. 核心API调用逻辑 ---

def call_vision_model(client: OpenAI, model: str, system_prompt: str, user_text: str, image_path: Path) -> str:
    """调用DashScope视觉模型 (非流式)。"""
    base64_image = encode_image_base64(image_path)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    try:
        print("vlm处理中...")
        completion = client.chat.completions.create(model=model, messages=messages)
        content = completion.choices[0].message.content
        print(content) # 直接打印完整结果
        return content
    except Exception as e:
        print(f"\n[API错误] 调用视觉模型失败: {e}")
        return "None" # 出错时返回None，避免中断流程

def stream_text_model_response(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    """调用DashScope文本模型 (流式)。"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    response_tokens = []
    try:
        completion = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                response_tokens.append(token)
        return "".join(response_tokens)
    except Exception as e:
        print(f"\n[API错误] 调用文本模型失败: {e}")
        return "" # 出错时返回空字符串

# --- 4. 阶段一：PDF表格提取 ---

def process_single_pdf_page(image_path: Path, page_num: int, total_pages: int, pdf_path: Path, client: OpenAI, config: dict, prompt: str):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"■■■ PDF: {pdf_path.name} | 页面: {page_num}/{total_pages} ■■■")
    user_text = "请根据图片内容和上述指令进行分析。"
    output_text = call_vision_model(client, config['vision_model'], prompt, user_text, image_path)
    if output_text.strip().lower() != "none":
        output_dir = config['output_dir'] / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        output_page_path = output_dir / f"{pdf_path.stem}_{page_num}.txt"
        output_page_path.write_text(output_text, encoding="utf-8")
        print(f"\n-> 结果已保存至: {output_page_path}")
    else:
        print(f"\n-> 无表格或API出错，跳过保存。")

def run_extraction_stage(config: dict, client: OpenAI):
    print("\n" + "="*60 + "\n--- 阶段一：开始PDF表格提取 ---\n" + "="*60)
    prompt = load_prompt(Path(__file__).parent / "Image2TextPrompt.txt")
    pdfs = sorted(list(config['pdf_source_dir'].glob("*.pdf")))
    if not pdfs: print(f"在 '{config['pdf_source_dir']}' 中未找到PDF文件。"); return
    pdfs_to_process = [p for p in pdfs if not (config['output_dir'] / p.stem).is_dir()]
    print(f"-> 跳过 {len(pdfs) - len(pdfs_to_process)} 个已处理的PDF。发现 {len(pdfs_to_process)} 个新文件。")
    if not pdfs_to_process: return
    for i, pdf_path in enumerate(pdfs_to_process):
        print(f"\n--- 开始处理第 {i+1}/{len(pdfs_to_process)} 个文件: {pdf_path.name} ---")
        try:
            images = pdf_to_images(pdf_path, config['image_cache_dir'])
            for j, image_path in enumerate(images):
                process_single_pdf_page(image_path, j + 1, len(images), pdf_path, client, config, prompt)
        except Exception as e: print(f"\n[错误] 处理 {pdf_path.name} 失败: {e}")

# --- 5. 阶段二：JSON自然语言序列化 ---

def serialize_single_file(source_path: Path, client: OpenAI, config: dict, prompt: str):
    content = source_path.read_text(encoding='utf-8').strip()
    if not content: print("-> 文件为空，跳过。"); return
    serialized_text = stream_text_model_response(client, config['text_model'], prompt, content)
    if serialized_text:
        relative_path = source_path.relative_to(config['output_dir'])
        target_path = config['serialization_output_dir'] / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(serialized_text, encoding='utf-8')
        print(f"\n-> 序列化结果已保存至: {target_path}")

def run_serialization_stage(config: dict, client: OpenAI):
    print("\n" + "="*60 + "\n--- 阶段二：开始JSON自然语言序列化 ---\n" + "="*60)
    prompt = load_prompt(Path(__file__).parent / "SerializationPrompt.txt")
    source_files = sorted(list(config['output_dir'].rglob("*.txt")))
    if not source_files: print(f"在 '{config['output_dir']}' 中未找到.txt文件。"); return
    files_to_process = [f for f in source_files if not (config['serialization_output_dir'] / f.relative_to(config['output_dir'])).exists()]
    print(f"-> 跳过 {len(source_files) - len(files_to_process)} 个已序列化的文件。发现 {len(files_to_process)} 个新文件。")
    if not files_to_process: return
    for i, file_path in enumerate(files_to_process):
        print(f"\n--- 开始序列化第 {i+1}/{len(files_to_process)} 个文件: {file_path} ---")
        try: serialize_single_file(file_path, client, config, prompt)
        except Exception as e: print(f"\n[错误] 序列化 {file_path} 失败: {e}")

# --- 6. 阶段三：合并输出文件 ---

def merge_files_in_directory(source_dir: Path, output_file_path: Path, file_type_desc: str):
    print(f"\n-> 正在合并 {file_type_desc} 文件...")
    files_to_merge = sorted(list(source_dir.rglob("*.txt")))
    if not files_to_merge: print(f"  -> 在 '{source_dir}' 中没有找到可合并的文件。"); return
    with output_file_path.open('w', encoding='utf-8') as outfile:
        for i, file_path in enumerate(files_to_merge):
            if i > 0: outfile.write("\n%%%%\n\n")
            header = f"-<This Chunk Is From {file_path.name}>-\n"
            content = file_path.read_text(encoding='utf-8').strip()
            outfile.write(header); outfile.write(content)
    print(f"  -> 合并完成！ {len(files_to_merge)} 个文件已合并至: {output_file_path}")

def run_combination_stage(config: dict):
    print("\n" + "="*60 + "\n--- 阶段三：开始合并输出文件 ---\n" + "="*60)
    combination_dir = config['combination_output_dir']
    combination_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merge_files_in_directory(config['output_dir'], combination_dir / f"combined_json+description_{timestamp}.txt", "JSON+Description")
    if config['serialization_enabled']:
        merge_files_in_directory(config['serialization_output_dir'], combination_dir / f"combined_natural_language_serial_{timestamp}.txt", "Natural Language")

# --- 7. 主执行流程 ---

def main():
    """脚本主入口函数"""
    print("--- PDF表格处理工具 (DashScope API模式) 启动 ---")
    try:
        config = setup_configuration(Path(__file__).parent / 'config.ini')
        
        for key in ['pdf_source_dir', 'image_cache_dir', 'output_dir', 'serialization_output_dir', 'combination_output_dir']:
            config[key].mkdir(exist_ok=True)
        
        # 初始化一个全局的API客户端
        client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
        print("-> 成功初始化DashScope API客户端。")
        
        # --- 执行阶段一 ---
        run_extraction_stage(config, client)

        # --- 执行阶段二 (如果启用) ---
        if config['serialization_enabled']:
            run_serialization_stage(config, client)
        
        # --- 执行阶段三 ---
        run_combination_stage(config)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n[致命错误] {e}")
    except Exception as e:
        print(f"\n[致命错误] 脚本启动失败，发生未知错误: {e}")
        
    print("\n--- 所有任务处理完毕 ---")

if __name__ == "__main__":
    main()
