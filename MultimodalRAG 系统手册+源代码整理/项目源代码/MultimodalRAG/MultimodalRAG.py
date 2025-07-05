# -------------------------------------------------------------------------------------------------
# 导入标准库模块
# 这是系统稳定运行的基础，我必须确保每一个模块都正确导入和使用。
# -------------------------------------------------------------------------------------------------
import sqlite3 # 导入 SQLite 数据库模块。它是我们存储和管理文档元数据的关键，比如文档的内部ID、原始ID、文本描述和图像路径。
import os      # 导入操作系统模块。它提供了与操作系统交互的必要功能，例如处理文件路径、检查文件或目录是否存在、以及创建目录等。这些操作对于管理索引和数据文件至关重要。
import numpy as np # 导入 NumPy 库。它提供了高效的数值计算能力，特别是在处理向量（如CLIP模型生成的特征向量）时不可或缺，能极大地提升性能。
from typing import List, Dict, Union, Optional, Tuple, Any # 导入类型提示模块。使用类型提示能让代码更清晰、更易于理解和维护，也能帮助静态分析工具发现潜在错误，这是高质量代码的重要保障。(增加了 Any 类型，以适应某些字典中可能包含的更广泛的数据类型)
import json    # 导入 JSON 库。用于处理 JSON (JavaScript Object Notation) 格式的数据，常用于配置文件读写、API数据交换等。我们的数据源就是JSON格式，正确处理它非常重要。
import time    # 导入时间库。提供时间相关的函数，比如获取当前时间、程序暂停（sleep）等，可以在需要时用于控制程序流程或添加延时。
import random  # 导入随机库。用于生成伪随机数，例如在示例查询中随机选择文档，以展示系统的多种能力。
import logging # 导入日志模块。这是追踪程序运行状态、诊断问题、记录信息、警告和错误的核心工具。详细的日志是确保系统可维护性的基石。
import sys     # 导入系统模块。提供了访问由 Python 解释器使用或维护的变量和函数的接口，此处用于配置日志输出到标准输出，方便实时监控。
import datetime # 导入日期时间模块。用于处理日期和时间，如此处用于生成带有时间戳的目录名，确保每次运行的输出结果能够唯一且易于组织。
import re      # 导入正则表达式模块。用于进行强大的文本模式匹配和字符串操作，如此处用于清理文件名中的非法字符，确保文件路径的有效性。

# -------------------------------------------------------------------------------------------------
# 导入第三方库模块 (这些是实现多模态功能的核心，需要预先安装。我已确认其必要性。)
# -------------------------------------------------------------------------------------------------
import faiss   # 导入 Faiss 库。这是一个由 Facebook AI Research 开发的、用于高效相似度搜索和聚类的向量库。它将是我们的向量检索引擎。
               # 安装提示: pip install faiss-cpu (如果您使用CPU) 或 pip install faiss-gpu (如果您有CUDA环境的GPU)。
from transformers import CLIPProcessor, CLIPModel # 从 Hugging Face Transformers 库导入 CLIP 模型的处理器和模型本身。
                                                 # CLIP (Contrastive Language–Image Pre-training) 是一个强大的多模态模型，能够将文本和图像编码到同一个向量空间，这是我们实现多模态检索的关键技术。
                                                 # 安装提示: pip install transformers torch pillow。
from PIL import Image, UnidentifiedImageError # 导入 Pillow 库 (PIL 的一个分支)。它是Python中事实上的图像处理标准库，用于图像文件的加载、处理和保存。 (增加了 UnidentifiedImageError 用于捕获特定的图像加载错误)
import torch   # 导入 PyTorch 库。这是一个广泛使用的开源机器学习框架，Transformers 库基于它构建。我们使用它来加载和运行CLIP模型，并利用其GPU加速能力（如果可用）。
import zhipuai # 导入 ZhipuAI 客户端库。用于与智谱 AI 开发的大语言模型 (LLM) API 进行交互。它将负责根据检索到的信息生成最终答案。
               # 安装提示: pip install zhipuai。

from MultimodalEncoder import MultimodalEncoder
from Indexer import Indexer
from Retriever import Retriever
from Generator import Generator

# -------------------------------------------------------------------------------------------------
# 全局日志记录器设置 (在 `if __name__ == "__main__":` 中会进一步精细配置，这里只是初始化)
# 这是一个重要的工具，我必须确保它随时可用，以便记录系统的每一个动作和潜在问题。
# -------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__) # 初始化一个模块级别的日志记录器实例。`__name__` 会被设置成当前模块的名称，便于区分日志来源。

# -------------------------------------------------------------------------------------------------
# 工具函数定义
# 这些是辅助性的功能，但同样重要，必须确保它们可靠无误。
# -------------------------------------------------------------------------------------------------
def setup_logging(log_file_path: str):
    """
    配置全局日志记录器 (logger)。
    此函数设定日志记录的最低级别、输出格式，并将日志信息同时发送到控制台和指定的日志文件。
    这是确保系统运行可追踪性的重要步骤。

    Args:
        log_file_path (str): 日志文件的完整路径。程序运行的所有日志信息将被精确地记录到此文件。
    """
    global logger # 声明我们要修改的是全局变量 `logger`。
    logger.setLevel(logging.INFO) # 设置日志记录的最低级别为 INFO。这意味着只有 INFO 及以上级别（如 WARNING, ERROR, CRITICAL）的日志才会被处理和输出。

    # 在添加新的处理器之前，清理可能已存在的旧处理器。这样做是为了避免在重复调用此函数时（例如在交互式环境或测试中）导致日志被多次记录。这是保证日志行为一致性的重要细节。
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个文件处理器 (FileHandler)，用于将日志信息写入到指定的日志文件。
    # `encoding='utf-8'` 确保日志文件能正确处理包括中文在内的多语言字符。
    # `mode='w'` 表示每次运行程序时会覆盖（overwrite）之前的日志文件，确保日志只记录当次运行的情况。如果需要保留历史日志，可以将模式改为 'a' (追加)。
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO) # 文件处理器也只处理 INFO 及以上级别的日志。

    # 创建一个控制台处理器 (StreamHandler)，用于将日志信息输出到标准输出（通常是终端控制台）。
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # 控制台处理器同样只处理 INFO 及以上级别的日志。

    # 定义日志格式器 (Formatter)。它决定了每条日志记录的显示格式。一个清晰的格式有助于快速定位问题。
    # 格式字符串包含:
    #   %(asctime)s: 日志记录的创建时间。
    #   %(levelname)s: 日志级别 (例如 INFO, WARNING, ERROR)。
    #   [%(filename)s.%(funcName)s:%(lineno)d]: 日志发出的文件名、函数名和行号。这是快速定位代码位置的关键信息。
    #   %(message)s: 实际的日志消息内容。
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s.%(funcName)s:%(lineno)d] - %(message)s')

    # 将定义好的格式器应用到文件处理器和控制台处理器。
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将配置好的文件处理器和控制台处理器添加到全局日志记录器中。
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("全局日志记录器配置完成。日志将同时输出到控制台，并写入文件: %s", log_file_path) # 记录一条日志，明确表明配置已成功。

def sanitize_filename(filename: str, max_length: int = 100, is_dir_component: bool = False) -> str:
    """
    清理输入的字符串，使其成为一个有效的文件名或目录名组件。
    这个函数会替换或移除文件名中可能导致问题的特殊字符，并将文件名截断到指定的最大长度，以确保跨操作系统的兼容性和文件系统的稳定性。

    Args:
        filename (str): 需要被清理的原始字符串。
        max_length (int): 清理后文件名的最大允许长度。默认为 100 个字符。
        is_dir_component (bool): (此参数在此当前的实现中未产生不同行为，保留以备未来扩展) 指示该字符串是否用作目录路径的一部分。
                                 为了简单和安全起见，此函数对文件名和目录名组件采用相同的严格清理规则。

    Returns:
        str: 清理和截断后的、可以用作文件系统名称的字符串。这个结果是可靠且安全的。
    """
    # 如果输入的文件名为空或 None，返回一个默认的占位符名称。这是为了避免生成空名称的文件或目录，增加系统的鲁棒性。
    if not filename:
        return "unnamed_component" # 未命名组件

    # 使用正则表达式替换掉文件名中常见的非法字符。
    # 这些字符 (\ / * ? : " < > |) 在大多数主流文件系统中都是不允许的。
    # 将这些非法字符统一替换为下划线 "_"。
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)

    # 将字符串两端的空白字符（空格、制表符、换行符等）去除。
    # 然后，将字符串内部的一个或多个连续空白字符替换为单个下划线 "_"。这能使文件名更紧凑和规范。
    sanitized = re.sub(r'\s+', '_', sanitized.strip())

    # 移除文件名开头可能存在的点和下划线。以点开头的在某些系统上是隐藏文件，以下划线开头的可能导致不规范。
    sanitized = re.sub(r'^[\._]+', '', sanitized)

    # 将清理后的字符串截断到 `max_length` 指定的最大长度。
    # 注意：简单的切片可能在多字节字符（如某些中文）的中间截断，导致乱码。对于本例主要处理ASCII文件名，这通常不是问题。若需完美处理，需要更复杂的截断逻辑，但当前实现已足够满足大多数常见文件名场景。
    sanitized = sanitized[:max_length]

    # 再次检查，如果清理和截断后字符串变为空，或者只包含点号 "." (可能导致隐藏文件或路径问题，尤其是在Unix-like系统中)，则返回一个特定的占位符名称。这是最后一层安全检查。
    if not sanitized or all(c == '.' for c in sanitized):
        return "sanitized_empty_name" # 清理后为空的名称

    # 避免使用 Windows 系统中的保留设备名作为文件名（不区分大小写）。
    # 例如: CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9。这些名称在某些操作下可能导致问题。
    # 如果清理后的名称（转换为大写后）匹配这些保留名，则在其前后添加下划线以作区分。这是一个简化的检查，完整的跨平台文件名验证会更复杂，但这个检查覆盖了常见的高风险情况。
    reserved_names_check = sanitized.upper()
    if reserved_names_check in ["CON", "PRN", "AUX", "NUL"] or \
       re.match(r"COM[1-9]$", reserved_names_check) or \
       re.match(r"LPT[1-9]$", reserved_names_check):
        sanitized = f"_{sanitized}_" # 在保留名称前后加下划线

    return sanitized # 返回最终清理后的、安全且符合文件系统规范的文件名字符串。

# -------------------------------------------------------------------------------------------------
# 数据加载与预处理模块
# 这是系统获取原始知识的基础步骤，必须确保数据的准确性和完整性。
# -------------------------------------------------------------------------------------------------
def load_data_from_json_and_associate_images(json_path: str, image_dir: str) -> List[Dict[str, Any]]:
    """
    从指定的 JSON 文件加载文档的元数据 (如 ID 和描述文本)，
    并根据文档 ID (JSON中的 'name' 字段) 在指定的图像目录中查找并关联对应的图像文件。
    函数假设图像文件名是文档 ID 加上常见的图片扩展名 (如 .png, .jpg)。

    Args:
        json_path (str): 包含文档元数据的 JSON 文件路径。
                         JSON 文件应为一个列表，其中每个对象至少包含 'name' 和 'description' 字段。
        image_dir (str): 存放与 JSON 数据对应的图片文件的目录路径。

    Returns:
        List[Dict[str, Any]]: 一个包含处理后文档信息的字典列表。
                    每个字典包含以下键：
                    - 'id': 文档的唯一标识符 (来自 JSON 'name' 字段，确保为字符串)。
                    - 'text': 文档的文本描述 (来自 JSON 'description' 字段，确保为字符串或 None)。
                    - 'image_path': 找到的对应图像文件的完整路径 (str)。如果未找到图像或 image_dir 无效，则为 None。
                    如果 JSON 文件不存在、无法解析或读取失败，则返回空列表 ([]).
    """
    # 获取当前模块的日志记录器实例，用于记录此函数的执行信息。详细的日志有助于追踪数据加载过程。
    func_logger = logging.getLogger(__name__) # 使用模块级 logger
    func_logger.info(f"开始从 JSON 文件 '{json_path}' 加载数据，并在目录 '{image_dir}' 中关联图像...")

    # 步骤 1: 检查 JSON 文件是否存在。如果文件不存在，这是个严重问题，必须立即报告并停止。
    if not os.path.exists(json_path):
        func_logger.error(f"错误：JSON 文件 '{json_path}' 未找到。请检查文件路径是否正确。")
        return [] # 文件不存在，无法继续，返回空列表。

    # 初始化用于存储最终处理后文档信息的列表。
    documents: List[Dict[str, Any]] = []
    # 定义一个包含常见图像文件扩展名的列表。我们必须考虑多种可能的图像格式。
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'] # 增加了 .webp 格式以支持更多现代格式

    # 步骤 2: 尝试打开并解析 JSON 文件。文件读取和解析是容易出错的地方，必须小心处理异常。
    try:
        # 使用 'with' 语句确保文件在使用后自动关闭，即使发生错误。这是良好的资源管理习惯。
        # 'r' 表示以只读模式打开文件。
        # 'encoding='utf-8'' 指定使用 UTF-8 编码读取文件，以正确处理包括中文在内的各种字符。
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f) # 解析 JSON 数据，将其转换为 Python 的列表或字典。
            # 验证 JSON 数据的顶层结构必须是列表。不符合预期的格式是数据源常见问题。
            if not isinstance(json_data, list):
                func_logger.error(f"错误: JSON 文件 '{json_path}' 的顶层结构不是一个列表。请确保JSON文件格式正确。")
                return []
    except json.JSONDecodeError as e:
        # 如果 JSON 文件内容格式不正确，json.load() 会抛出 JSONDecodeError。这是 JSON 数据格式问题的明确指示。
        func_logger.error(f"错误：JSON 文件 '{json_path}' 解析失败。错误详情: {e}")
        func_logger.error(f"        请确保文件内容是有效的 JSON 格式 (一个包含对象的列表)。")
        return [] # JSON 格式错误，返回空列表。
    except Exception as e:
        # 捕获其他可能的读取文件错误，例如文件权限问题或磁盘问题。必须考虑到所有可能性。
        func_logger.error(f"错误：读取 JSON 文件 '{json_path}' 时发生未知错误。错误详情: {e}")
        return [] # 其他读取错误，返回空列表。

    func_logger.info(f"已成功从 '{json_path}' 加载 {len(json_data)} 条原始记录。")

    # 初始化计数器，用于统计数据处理过程中的情况。清晰的统计数据有助于了解数据质量。
    found_images_count = 0    # 成功关联到图像的文档数量。
    missing_key_count = 0     # 因缺少必要字段 ('name' 或 'description') 或内容无效而被跳过的记录数量。
    image_dir_warning_issued = False # 用于控制图像目录无效警告只输出一次。

    # 步骤 3: 遍历从 JSON 文件加载的每一条原始记录。这是核心的处理循环。
    for item_index, item in enumerate(json_data): # 使用 enumerate 获取索引，方便日志记录和问题定位。
        # 确保每一项都是字典。非字典项是无效数据。
        if not isinstance(item, dict):
            func_logger.warning(f"警告：跳过第 {item_index + 1} 条记录（JSON索引 {item_index}），因其不是一个有效的字典对象。记录内容: {item}")
            missing_key_count += 1
            continue

        doc_id = item.get('name')         # 尝试获取 'name' 字段作为文档 ID。使用 .get() 是安全的，即使键不存在也不会引发错误。
        text_content = item.get('description') # 尝试获取 'description' 字段作为文本内容。

        # 检查关键字段 'name' 和 'description' 是否存在且有值。这两个字段是文档的基本信息，缺失则无法索引。
        # 如果任一字段缺失或为空字符串（去除空白后），则跳过该条记录。
        valid_doc_id = doc_id is not None and str(doc_id).strip()
        valid_text = text_content is not None and str(text_content).strip()

        if not valid_doc_id or not valid_text:
            missing_key_count += 1
            reason = []
            if not valid_doc_id: reason.append("'name'字段缺失或为空")
            if not valid_text: reason.append("'description'字段缺失或为空")
            # 记录详细警告，包括原始索引和部分内容，便于用户排查数据源。
            func_logger.warning(f"警告：跳过第 {item_index + 1} 条记录（原始JSON索引 {item_index}），原因: {', '.join(reason)}。记录内容: {item}")
            continue # 继续处理下一条记录。

        # 初始化图像路径为 None。如果在指定目录中找不到匹配的图像，它将保持为 None。
        image_path: Optional[str] = None
        # 检查图像目录路径是否有效（已提供且存在于文件系统中）。只有目录有效时才尝试查找图像。
        if image_dir and os.path.isdir(image_dir): # 确保 image_dir 是一个存在的目录
            # 遍历预定义的图像扩展名列表，尝试构建并查找图像文件。
            for ext in image_extensions:
                # 构建潜在的图像文件名：文档ID（来自 'name' 字段）+ 当前扩展名。
                # 使用 str(doc_id) 确保即使 doc_id 是数字也能正确拼接。
                potential_image_filename = str(doc_id) + ext
                # 使用 os.path.join 安全地构建跨平台的完整图像文件路径。
                potential_image_path = os.path.join(image_dir, potential_image_filename)

                # 检查构建的图像文件路径是否存在于文件系统中，并且是一个文件。
                if os.path.exists(potential_image_path) and os.path.isfile(potential_image_path): # 确保是文件
                    image_path = potential_image_path # 找到图像，记录其完整路径。
                    found_images_count += 1           # 增加找到图像的计数。
                    break # 找到一个匹配的图像后，无需再检查其他扩展名，跳出内层循环。
        elif image_dir and not os.path.isdir(image_dir) and not image_dir_warning_issued:
            # 如果提供了 image_dir 但它不是一个有效的目录，记录一次警告。使用标志避免重复警告。
            func_logger.warning(f"提供的图像目录 '{image_dir}' 不是一个有效的目录，将无法关联图像。")
            image_dir_warning_issued = True # 设置标志，表示警告已发出。
        elif not image_dir:
             # 如果未提供图像目录，在 DEBUG 级别记录，避免在正常运行时输出过多信息。
             func_logger.debug(f"未提供图像目录 (image_dir 为 None 或空)，将不尝试关联图像。")


        # 将处理后的文档信息（包括 ID、文本和可能的图像路径）添加到 `documents` 列表中。
        documents.append({
            'id': str(doc_id), # 确保文档 ID 是字符串类型，以便后续一致处理。
            'text': str(text_content) if text_content is not None else None, # 确保文本是字符串；如果原始为 None，则保持 None。
            'image_path': image_path # 存储找到的图像路径，如果未找到则为 None。
        })

    # 步骤 4: 打印数据加载和关联过程的总结信息。清晰的总结是流程结束的标志。
    func_logger.info(f"成功准备了 {len(documents)} 个文档用于后续处理。")
    if missing_key_count > 0:
        func_logger.warning(f"在原始 JSON 数据中，共有 {missing_key_count} 条记录因格式无效或缺少有效 'name'/'description' 字段而被跳过。")
    func_logger.info(f"在有效文档中，共有 {found_images_count} 个文档成功关联了图像文件。")

    # 如果指定了图像目录，但没有找到任何图像文件（并且至少有一个文档被处理了），则给出提示，帮助用户排查图片关联问题。
    if len(documents) > 0 and found_images_count == 0 and image_dir and os.path.isdir(image_dir):
         func_logger.info(f"提示: 未在目录 '{image_dir}' 中找到任何与文档 ID 匹配的图像文件。")
         func_logger.info(f"        请检查图像文件名是否严格遵循 '文档ID.扩展名' 的格式 (例如，如果文档 'name' 是 'item01'，则图像应为 'item01.png')。")

    func_logger.info(f"--- 数据加载与图像关联流程结束 ---")
    return documents # 返回包含所有已处理文档信息的列表。

# -------------------------------------------------------------------------------------------------
# 主程序执行入口 (示例使用流程)
# This is an end-to-end usage example demonstrating how to initialize components and execute the RAG flow.
# I will ensure each step is clear, with appropriate error handling and logging.
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # =============================================================================================
    # Step 0: Configure Run Parameters and Output Directory (Important! Modify as needed!)
    # This is a critical setup phase. Ensure all paths and configurations are correct.
    # =============================================================================================

    # --- User-configurable Run Identifier (Used as the fixed top-level output directory name) ---
    # Set a meaningful descriptive name for this run, e.g., project name or system identifier.
    # This name, after sanitization, will be used directly as the top-level output directory name.
    # Note: Running the script multiple times with the same identifier will overwrite content in the output directory.
    RUN_IDENTIFIER_BASE: str = "multimodal_rag_system_output" # Example: Fixed base name for the output directory

    # Sanitize the run identifier to ensure it's a valid directory name.
    sanitized_run_identifier: str = sanitize_filename(RUN_IDENTIFIER_BASE, max_length=50)

    # --- Construct the Fixed Top-Level Output Directory ---
    # The directory name now directly uses the sanitized run identifier, without a timestamp.
    OUTPUT_BASE_DIR: str = sanitized_run_identifier

    # Create the output directory (exist_ok=True prevents errors if it already exists).
    # Subsequent runs will reuse this directory.
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # --- Configure Logging ---
    # Log files will now be located in this fixed top-level directory under the 'logs' subdirectory.
    # Note: The log file mode is still 'w' (write/overwrite), overwriting the old log file on each run.
    # If you need to append logs, change mode='w' to mode='a' in the setup_logging function.
    LOG_DIR: str = os.path.join(OUTPUT_BASE_DIR, "logs") # Log subdirectory name (English)
    os.makedirs(LOG_DIR, exist_ok=True)
    # Define the full path for the log file.
    LOG_FILE_PATH: str = os.path.join(LOG_DIR, "system_execution_log.txt") # Log file name (English)
    setup_logging(LOG_FILE_PATH) # Call the logging setup function.

    logger.info("\n" + "="*80)
    logger.info("========= Multimodal Retrieval-Augmented Generation (RAG) System =========") # English title
    logger.info("=========                     Main Execution Start                   =========") # English title
    logger.info("="*80 + "\n")
    logger.info(f"User-defined run identifier (used as fixed directory name): {RUN_IDENTIFIER_BASE}")
    logger.info(f"Sanitized run identifier (final directory name): {sanitized_run_identifier}")
    logger.info(f"All output data will be saved to the fixed top-level directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
    logger.warning("NOTE: This top-level output directory name is fixed. Subsequent runs with the same identifier will OVERWRITE content in this directory (including logs, database, indices, and query results).") # English warning

    # --- Data Source Configuration ---
    # Source of the input data. Ensure paths are correct.
    JSON_DATA_PATH: str = 'data.json'
    IMAGE_DIR_PATH: str = 'images'
    logger.info(f"Data source config: JSON metadata file='{JSON_DATA_PATH}', Image directory='{IMAGE_DIR_PATH}'")

    # --- Persistence Storage File Paths (within the fixed top-level directory) ---
    # Database and Faiss indices need persistence for reloading on next runs. Define paths explicitly.
    DB_STORAGE_DIR: str = os.path.join(OUTPUT_BASE_DIR, "data_storage") # Main data storage directory
    DB_DIR: str = os.path.join(DB_STORAGE_DIR, "database") # Database subdirectory
    DB_FILE: str = os.path.join(DB_DIR, 'multimodal_doc_store.db') # Database file name (English)

    FAISS_DIR: str = os.path.join(DB_STORAGE_DIR, "vector_indices") # Faiss index subdirectory (English)
    FAISS_TEXT_INDEX_FILE: str = os.path.join(FAISS_DIR, 'text_vector_index.faiss') # Text index file name (English)
    FAISS_IMAGE_INDEX_FILE: str = os.path.join(FAISS_DIR, 'image_vector_index.faiss') # Image index file name (English)
    FAISS_MEAN_INDEX_FILE: str = os.path.join(FAISS_DIR, 'mean_vector_index.faiss') # Mean index file name (English)

    # Query Results Output Directory (within the fixed top-level directory)
    # Used to save detailed input, retrieval results, and LLM generation results for each query.
    QUERY_RESULTS_DIR: str = os.path.join(OUTPUT_BASE_DIR, "query_session_results") # English name
    os.makedirs(QUERY_RESULTS_DIR, exist_ok=True)

    logger.info(f"Database file will be saved to: {DB_FILE}")
    logger.info(f"Faiss index files will be saved to directory: {FAISS_DIR}")
    logger.info(f"Query session results will be saved to directory: {QUERY_RESULTS_DIR}")


    # --- Model Configuration ---
    # Choosing appropriate models is crucial for system performance. Using models specified in the example.
    # Note: The following models were chosen deliberately to balance performance and resource consumption.
    # CLIP Model: "openai/clip-vit-base-patch32" is a widely used baseline model.
    # LLM Model: "glm-4-flash" from ZhipuAI is a lighter model suitable for faster responses and resource-constrained scenarios.
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    LLM_MODEL: str = "glm-4-flash"
    logger.info(f"Model configuration: CLIP Model='{CLIP_MODEL}', Large Language Model (LLM)='{LLM_MODEL}'")

    # =============================================================================================
    # Step 1: Load Raw Data and Attempt to Associate Images
    # First step in building the knowledge base.
    # =============================================================================================
    logger.info("\n--- [Main Flow] Step 1: Load document data from JSON and associate image files ---")
    documents_to_index: List[Dict[str, Any]] = load_data_from_json_and_associate_images(JSON_DATA_PATH, IMAGE_DIR_PATH)

    # Check if data loading was successful. Subsequent steps depend on this.
    if not documents_to_index:
        logger.critical("CRITICAL ERROR: Failed to load any valid document data from the JSON file, or the document list is empty after image association.") # English error
        logger.critical(f"          Please check if file '{JSON_DATA_PATH}' exists, is correctly formatted, and contains valid records.") # English error
        logger.critical("          Program will exit now due to data loading failure.") # English error
        exit(1) # Exit because necessary data is missing.
    logger.info(f"--- [Main Flow] Step 1 Complete: Successfully loaded and prepared {len(documents_to_index)} documents for indexing. ---\n")
    time.sleep(0.2) # Short pause for clearer log output.

    # =============================================================================================
    # Step 2: Initialize Indexer and Build Index for Loaded Documents
    # Key step to convert raw data into vectors and store in DB and Faiss indices.
    # =============================================================================================
    logger.info("--- [Main Flow] Step 2: Initialize Indexer and build index for loaded documents ---")
    indexer_instance: Optional[Indexer] = None # Initialize Indexer instance variable.
    try:
        # Initialize Indexer, passing all necessary paths and the model name.
        indexer_instance = Indexer(
            db_path=DB_FILE,
            faiss_text_index_path=FAISS_TEXT_INDEX_FILE,
            faiss_image_index_path=FAISS_IMAGE_INDEX_FILE,
            faiss_mean_index_path=FAISS_MEAN_INDEX_FILE,
            clip_model_name=CLIP_MODEL
        )
        # Call index_documents method to start the indexing process.
        indexer_instance.index_documents(documents_to_index)

        logger.info("Index building/loading complete. Current status:") # English status
        # Report current data counts in the database and Faiss indices.
        text_count = getattr(indexer_instance.text_index, 'ntotal', 0)
        image_count = getattr(indexer_instance.image_index, 'ntotal', 0)
        mean_count = getattr(indexer_instance.mean_index, 'ntotal', 0)
        db_doc_count = indexer_instance.get_document_count()
        logger.info(f"  - SQLite Database ('{os.path.basename(DB_FILE)}') document records: {db_doc_count}") # English label
        logger.info(f"  - Text Faiss Index ('{os.path.basename(FAISS_TEXT_INDEX_FILE)}') vectors: {text_count}") # English label
        logger.info(f"  - Image Faiss Index ('{os.path.basename(FAISS_IMAGE_INDEX_FILE)}') vectors: {image_count}") # English label
        logger.info(f"  - Mean Faiss Index ('{os.path.basename(FAISS_MEAN_INDEX_FILE)}') vectors: {mean_count}") # English label

        # Check index status, warn if indices are empty but DB has records.
        if text_count == 0 and image_count == 0 and mean_count == 0 and db_doc_count > 0:
             logger.warning("WARNING: Database contains document records, but all Faiss indices are empty!") # English warning
             logger.warning("      This might indicate that the encoding process failed for all documents, or no valid vectors were generated.") # English warning
             logger.warning("      Subsequent retrieval operations will not be able to return results based on vector similarity. Please review Indexer and Encoder logs carefully.") # English warning
        elif db_doc_count == 0:
             logger.warning("WARNING: Database and all Faiss indices are currently empty.") # English warning
             logger.warning("      This could be because the input JSON data was empty, or all entries were skipped during loading and processing.") # English warning

    except Exception as e:
         logger.critical(f"CRITICAL ERROR: Top-level exception occurred during Indexer initialization or index building: {e}", exc_info=True) # English error
         logger.critical("          Subsequent retrieval and generation steps may not work correctly as the Indexer failed to prepare.") # English error
         indexer_instance = None # Set instance to None to indicate initialization failure.

    logger.info("--- [Main Flow] Step 2 Complete. ---\n")
    time.sleep(0.2)

    # =============================================================================================
    # Step 3: Initialize Retriever
    # Retriever depends on the Indexer. Initialize only if Indexer is successful and contains searchable vectors.
    # =============================================================================================
    logger.info("--- [Main Flow] Step 3: Initialize Retriever ---")
    retriever_instance: Optional[Retriever] = None # Initialize Retriever instance variable.
    # Attempt to initialize Retriever only if Indexer initialized successfully AND at least one Faiss index contains vectors.
    if indexer_instance: # Check if indexer object exists first
         text_index_ready = hasattr(indexer_instance, 'text_index') and getattr(indexer_instance.text_index, 'ntotal', 0) > 0
         image_index_ready = hasattr(indexer_instance, 'image_index') and getattr(indexer_instance.image_index, 'ntotal', 0) > 0
         mean_index_ready = hasattr(indexer_instance, 'mean_index') and getattr(indexer_instance.mean_index, 'ntotal', 0) > 0

         if text_index_ready or image_index_ready or mean_index_ready:
             try:
                 retriever_instance = Retriever(indexer=indexer_instance) # Pass the Indexer instance.
             except Exception as e:
                  logger.error(f"ERROR: Exception occurred during Retriever initialization: {e}", exc_info=True) # English error
                  retriever_instance = None # Initialization failed.
         else:
              logger.warning("Skipping Retriever initialization.") # English warning
              logger.warning("  Reason: Indexer initialized successfully, but all its Faiss indices are currently empty.") # English reason
              logger.warning("        This might be due to encoding issues, data problems, or index building logic. Please check detailed logs from Step 2.") # English reason
              logger.warning("        Retriever cannot perform effective operations without searchable vectors.") # English reason
    else:
         logger.warning("Skipping Retriever initialization.") # English warning
         logger.warning("  Reason: Indexer initialization failed. Please check logs for Step 2 (Indexer initialization and indexing).") # English reason


    logger.info("--- [Main Flow] Step 3 Complete. ---\n")
    time.sleep(0.2)

    # =============================================================================================
    # Step 4: Initialize Generator
    # Generator depends on the ZhipuAI API. Check if the API Key is available.
    # =============================================================================================
    logger.info("--- [Main Flow] Step 4: Initialize Generator (will interact with ZhipuAI API) ---")
    generator_instance: Optional[Generator] = None # Initialize Generator instance variable.
    # Try to get ZhipuAI API Key from environment variable (recommended).
    zhipuai_api_key_from_env: Optional[str] = os.getenv("ZHIPUAI_API_KEY")
    if not zhipuai_api_key_from_env:
        # If API Key is not found in environment, log a warning and explain how to set it.
        logger.warning("WARNING: Environment variable 'ZHIPUAI_API_KEY' not found.") # English warning
        logger.warning("      The Generator will not be able to communicate with the ZhipuAI API, and the answer generation step will be skipped.") # English warning
        logger.warning("      To enable LLM answer generation, please do one of the following:") # English instruction
        logger.warning("        1. (Recommended) Set your ZhipuAI API Key as an environment variable named 'ZHIPUAI_API_KEY'.") # English instruction
        logger.warning("           Example (Linux/macOS): export ZHIPUAI_API_KEY='your_valid_api_key'") # English example
        logger.warning("           Then, rerun this script in the same terminal session.") # English instruction
        logger.warning("        2. (Alternative) Pass the API Key directly via the `api_key` parameter when initializing the Generator in the code (less secure for this example).") # English instruction
    else:
        logger.info("Environment variable 'ZHIPUAI_API_KEY' detected. Attempting to initialize Generator...") # English info
        try:
            # Initialize Generator using the obtained API Key.
            generator_instance = Generator(api_key=zhipuai_api_key_from_env, model_name=LLM_MODEL)
        except Exception as e:
             logger.error(f"ERROR: Exception occurred during Generator initialization: {e}", exc_info=True) # English error
             generator_instance = None # Initialization failed.

    logger.info("--- [Main Flow] Step 4 Complete. ---\n")
    time.sleep(0.2)

    # =============================================================================================
    # Step 5: Execute RAG Query Examples (Retrieval + Generation stages)
    # End-to-end demonstration of the system. Executes only if both Retriever and Generator are available.
    # =============================================================================================
    logger.info("--- [Main Flow] Step 5: Execute RAG Query Examples (Retrieve + Generate) ---")

    # Execute query examples only if both Retriever and Generator initialized successfully.
    if retriever_instance and generator_instance:
        logger.info("Retriever and Generator are both successfully initialized. Proceeding with example queries...") # English info

        def log_retrieved_docs_summary_for_main_process(docs_list: List[Dict[str, Any]], query_log_prefix: str = "    "):
            """Helper function to print a concise summary of retrieved documents in the main process log."""
            if not docs_list:
                logger.info(f"{query_log_prefix}>> Retrieval Result: No relevant documents found for the query.") # English result
                return
            logger.info(f"{query_log_prefix}>> Retrieval Result: Found Top-{len(docs_list)} relevant documents. Summary:") # English result
            for i, doc_item_data in enumerate(docs_list):
                score = doc_item_data.get('score', 'N/A') # Get score.
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score) # Format score.
                text_preview = doc_item_data.get('text', 'No text content')[:70] # Truncate text preview.
                if len(doc_item_data.get('text', '')) > 70: text_preview += "..." # Add ellipsis.
                img_filename_info = "" # Initialize image info string.
                if doc_item_data.get('image_path'):
                    img_filename_info = f", Associated Image: '{os.path.basename(doc_item_data['image_path'])}'" # Add image filename.
                logger.info(f"{query_log_prefix}  {i+1}. Document ID: {doc_item_data.get('id', 'N/A')} (Score: {score_str})") # English output
                logger.info(f"{query_log_prefix}     Text Preview: '{text_preview}'{img_filename_info}") # English output
            logger.info(f"{query_log_prefix}{'-'*40}")

        # --- Prepare Example Query Data (Reduced quantity for resource limits, keep only a few examples) ---
        text_queries_examples: List[str] = [
            "What is a bandgap voltage reference and its main purpose?", # Example: Pure text query, concept explanation
            "Explain how the PTAT current is generated and its role in a bandgap circuit.", # Example: Pure text query, principle
        ]

        # Collect documents with valid images to build image and multimodal query examples.
        image_docs_available_for_queries: List[Dict[str, Any]] = []
        if documents_to_index:
            for doc_data_item_source in documents_to_index:
                img_path_source = doc_data_item_source.get('image_path')
                if img_path_source and os.path.exists(img_path_source) and os.path.isfile(img_path_source):
                    # Only add documents with valid image paths.
                    image_docs_available_for_queries.append({
                        'id': doc_data_item_source.get('id'),
                        'image_path': img_path_source,
                        'text': doc_data_item_source.get('text', '')
                    })

        image_queries_examples_data: List[Dict[str, Any]] = []
        multimodal_queries_examples_data: List[Dict[str, Any]] = []

        if image_docs_available_for_queries:
            # Randomly select a small number from available image docs for query examples.
            num_image_query_samples = min(1, len(image_docs_available_for_queries)) # Reduced to only 1 sample for image/multimodal query examples
            logger.info(f"Found {len(image_docs_available_for_queries)} documents with valid images. Randomly selecting {num_image_query_samples} for image/multimodal query examples.") # English info
            selected_image_docs_for_queries = random.sample(image_docs_available_for_queries, num_image_query_samples)

            for selected_doc_info in selected_image_docs_for_queries:
                doc_id_for_query = selected_doc_info['id']
                img_path_for_query = selected_doc_info['image_path']
                img_filename_for_query = os.path.basename(img_path_for_query)

                # Build pure image query example.
                image_queries_examples_data.append({
                    'query_input': {'image_path': img_path_for_query},
                    # Explicitly mention image info and requirements in the text question for Generator, guiding it based on text description.
                    'query_for_generator': f"What circuit structure or key concept does this image (filename: {img_filename_for_query}) primarily show? Please explain in detail based on the text description in the associated document.", # English query
                    'description': f"PureImageQuery_About_{img_filename_for_query}" # Description for logs and filenames (English)
                })

                # Build multimodal query example.
                multimodal_queries_examples_data.append({
                    'query_input': {
                        'text': f"Combining the document content and this image (filename: {img_filename_for_query}), please explain the working principle, key features, or design considerations of the circuit shown.", # English text query part
                        'image_path': img_path_for_query
                    },
                     # Text question for Generator can be similar to the text part of query_input, or adjusted as needed.
                    'query_for_generator': f"Combining the document content and this image (filename: {img_filename_for_query}), please explain the working principle, key features, or design considerations of the circuit shown.", # English query for generator
                    'description': f"MultimodalQuery_ExplainImage_{img_filename_for_query}" # Description for logs and filenames (English)
                })
        else:
             logger.warning("WARNING: No valid and existing image files found in the loaded data.") # English warning
             logger.warning("      Therefore, examples for pure image queries and multimodal queries will be skipped.") # English warning


        # Group all example queries by type.
        all_example_queries_groups: List[Tuple[str, List[Any]]] = [ # Adjusted type for List[Any]
            ("Pure Text Query", text_queries_examples), # English type name
            ("Pure Image Query", image_queries_examples_data), # English type name
            ("Multimodal Query", multimodal_queries_examples_data) # English type name
        ]

        overall_query_counter = 0 # Track total query count for unique output directories.
        # Iterate through each query group and each query within the group.
        for query_group_name, queries_in_group in all_example_queries_groups:
            logger.info(f"\n{'#'*70}\n>>> Starting Example Queries, Type: [{query_group_name}] (Total in this group: {len(queries_in_group)}) <<<\n{'#'*70}\n") # English header

            if not queries_in_group:
                logger.info(f"    (Skipping query examples of type [{query_group_name}] as no query data is available.)") # English skip message
                continue

            for query_index_in_group, query_data_item in enumerate(queries_in_group):
                overall_query_counter += 1

                # Prepare query input format for Retriever and text question for Generator.
                query_input_for_retriever: Union[str, Dict[str, str], None] = None
                query_text_for_generator: Optional[str] = None
                query_description_for_logging: Optional[str] = None     # Query description for logging and reports.
                query_file_prefix_for_saving: Optional[str] = None     # Directory name prefix for saving results.

                # Extract specific query data based on the group type.
                if query_group_name == "Pure Text Query":
                    query_input_for_retriever = str(query_data_item) # Pure text input is the string itself.
                    query_text_for_generator = str(query_data_item) # Generator also uses this string directly.
                    query_description_for_logging = str(query_data_item) # Log description.
                    query_file_prefix_for_saving = sanitize_filename(f"TextQuery_{query_data_item}", max_length=60) # Generate directory prefix.
                else: # Pure Image or Multimodal query, data item is a dictionary.
                    if isinstance(query_data_item, dict): # Add check to ensure it's a dict
                        query_input_for_retriever = query_data_item.get('query_input')
                        query_text_for_generator = query_data_item.get('query_for_generator')
                        # Use the predefined description, which should be a string.
                        query_description_for_logging = query_data_item.get('description')
                        if isinstance(query_description_for_logging, str):
                            query_file_prefix_for_saving = sanitize_filename(query_description_for_logging, max_length=60)
                        else:
                             query_file_prefix_for_saving = sanitize_filename(f"{query_group_name}_query_{query_index_in_group+1}", max_length=60) # Fallback prefix
                             logger.warning(f"Query #{overall_query_counter} description is not a string, using fallback filename prefix.")
                    else:
                        logger.error(f"Error processing query #{overall_query_counter}: Expected dictionary for {query_group_name}, but got {type(query_data_item)}. Skipping query.")
                        continue # Skip this invalid query item


                # Log information about the query currently being processed.
                logger.info(f"\n--- Processing Query #{overall_query_counter} (Type: {query_group_name} - Index in group: {query_index_in_group+1}/{len(queries_in_group)}) ---") # English status
                log_desc_str = str(query_description_for_logging) if query_description_for_logging else "N/A"
                logger.info(f"Query Description: {log_desc_str[:120]}{'...' if len(log_desc_str)>120 else ''}") # English label
                # Log the input being passed to the Retriever in detail.
                if isinstance(query_input_for_retriever, dict):
                    retriever_input_text = query_input_for_retriever.get('text')
                    retriever_input_image = query_input_for_retriever.get('image_path')
                    if retriever_input_text:
                        logger.info(f"  -> Input Text for Retriever: '{str(retriever_input_text)[:80]}{'...' if len(str(retriever_input_text)) > 80 else ''}'") # English label
                    if retriever_input_image:
                         logger.info(f"  -> Input Image for Retriever: '{os.path.basename(str(retriever_input_image))}'") # English label
                elif isinstance(query_input_for_retriever, str):
                     logger.info(f"  -> Input Text for Retriever: '{query_input_for_retriever[:80]}{'...' if len(query_input_for_retriever) > 80 else ''}'") # English label
                else:
                     logger.info("  -> Input for Retriever: (None or invalid format)")

                # Log the text question being passed to the Generator.
                if query_text_for_generator:
                     logger.info(f"  -> Question Text for Generator: '{str(query_text_for_generator)[:100]}{'...' if len(str(query_text_for_generator)) > 100 else ''}'") # English label
                else:
                     logger.info("  -> Question Text for Generator: (None)") # English label
                logger.info("-" * 30)

                # Create a unique output directory for the current query to save detailed results.
                # Ensure query_file_prefix_for_saving is set
                if not query_file_prefix_for_saving:
                    query_file_prefix_for_saving = sanitize_filename(f"query_{overall_query_counter}_fallback", max_length=60)
                    logger.warning(f"Query #{overall_query_counter} had no valid file prefix, using fallback: {query_file_prefix_for_saving}")

                current_query_specific_output_dir = os.path.join(QUERY_RESULTS_DIR, f"query_{overall_query_counter:03d}_{query_file_prefix_for_saving}")
                try:
                    os.makedirs(current_query_specific_output_dir, exist_ok=True)
                    logger.info(f"  Detailed results for this query will be saved to: {current_query_specific_output_dir}") # English info
                except OSError as e_mkdir:
                     logger.error(f"Error creating output directory for query #{overall_query_counter}: {e_mkdir}. Skipping saving results for this query.")
                     continue # Skip to next query if directory cannot be created


                retrieved_context_docs_list: List[Dict[str, Any]] = [] # Initialize retrieval results list.
                final_generated_response_text: str = "LLM generation step was not executed or failed due to an error." # Default generator failure message (English)

                try:
                    # Save the original query input passed to the Retriever.
                    query_input_filename = "input_for_retriever.json" if isinstance(query_input_for_retriever, dict) else "input_for_retriever.txt" # English filename
                    query_input_save_path = os.path.join(current_query_specific_output_dir, query_input_filename)
                    try:
                        with open(query_input_save_path, 'w', encoding='utf-8') as f_query_in:
                            if isinstance(query_input_for_retriever, dict):
                                json.dump(query_input_for_retriever, f_query_in, ensure_ascii=False, indent=4)
                            else:
                                f_query_in.write(str(query_input_for_retriever) if query_input_for_retriever is not None else "")
                        logger.debug(f"  Query input saved to: {query_input_save_path}")
                    except Exception as e_save_input:
                         logger.error(f"Error saving query input to {query_input_save_path}: {e_save_input}")


                    logger.info("  [Retrieval Stage] Calling Retriever.retrieve() method...") # English stage label
                    # Ensure query_input_for_retriever is not None before calling retrieve
                    if query_input_for_retriever is not None:
                        # Call Retriever to perform retrieval. Use k=2 for Top-2 results for quicker demo.
                        retrieved_context_docs_list = retriever_instance.retrieve(query_input_for_retriever, k=2)
                        # Print summary of retrieved documents.
                        log_retrieved_docs_summary_for_main_process(retrieved_context_docs_list, query_log_prefix="    ")
                    else:
                        logger.warning("  [Retrieval Stage] Skipped: Query input for retriever was None or invalid.")
                        retrieved_context_docs_list = [] # Ensure it's empty


                    # Save the full list of retrieved documents (context).
                    retrieved_context_save_path = os.path.join(current_query_specific_output_dir, "retrieved_context_documents.json") # English filename
                    try:
                        with open(retrieved_context_save_path, 'w', encoding='utf-8') as f_retrieved_ctx:
                            json.dump(retrieved_context_docs_list, f_retrieved_ctx, ensure_ascii=False, indent=4)
                        logger.debug(f"  Full retrieved context documents saved to: {retrieved_context_save_path}")
                    except Exception as e_save_context:
                         logger.error(f"Error saving retrieved context to {retrieved_context_save_path}: {e_save_context}")


                    # Proceed to generation stage only if at least one document was retrieved.
                    if retrieved_context_docs_list:
                        logger.info("  [Generation Stage] Calling Generator.generate() method (using retrieved context)...") # English stage label
                        if query_text_for_generator:
                            # Call Generator to produce the final answer.
                            final_generated_response_text = generator_instance.generate(query_text_for_generator, retrieved_context_docs_list)

                            # Print the final response generated by the LLM.
                            logger.info(f"\n  <<< Final Response Generated by LLM for Query #{overall_query_counter} >>>") # English header
                            logger.info("-" * 35)
                            logger.info(final_generated_response_text)
                            logger.info("-" * 35)
                        else:
                            logger.error("  [Generation Stage] ERROR: Question text for Generator is None. Cannot generate response.") # English error
                            final_generated_response_text = "Error: Question text for the generator was empty." # English error message
                    else:
                         # If no context was retrieved, skip generation and log it.
                         logger.info("  [Generation Stage] Skipped: Retriever did not find any relevant context documents, so LLM generation is not performed.") # English skip message
                         final_generated_response_text = "No relevant context found by Retriever, LLM generation skipped." # English message

                    # Save the final response generated by the LLM.
                    llm_response_save_path = os.path.join(current_query_specific_output_dir, "llm_generated_final_response.txt") # English filename
                    try:
                        with open(llm_response_save_path, 'w', encoding='utf-8') as f_llm_resp:
                            f_llm_resp.write(final_generated_response_text)
                        logger.debug(f"  LLM generated response saved to: {llm_response_save_path}")
                    except Exception as e_save_response:
                         logger.error(f"Error saving LLM response to {llm_response_save_path}: {e_save_response}")


                except Exception as e_query_processing:
                     # Catch any exceptions during the processing of a single query, log as critical, but don't stop the whole program.
                     log_desc_str = str(query_description_for_logging) if query_description_for_logging else f"Query_{overall_query_counter}"
                     logger.critical(f"CRITICAL ERROR occurred while processing query '{log_desc_str}' (Query #{overall_query_counter}): {e_query_processing}", exc_info=True) # English error
                     # Try to save the error message to a file as well.
                     try:
                        error_info_path = os.path.join(current_query_specific_output_dir, "processing_error_info.txt") # English filename
                        with open(error_info_path, 'w', encoding='utf-8') as f_proc_err:
                            f_proc_err.write(f"A critical error occurred while processing this query: {e_query_processing}\nCheck the main log file for the detailed stack trace.\n\nOriginal query description: {log_desc_str}\n")
                     except Exception as e_save_err:
                        logger.error(f"Additional Error: Failed to save query processing error information to file: {e_save_err}") # English error

                logger.info(f"--- Query #{overall_query_counter} Processing Complete ---") # English status
                # Pause briefly before processing the next query in the same group, for easier log observation.
                if query_index_in_group < len(queries_in_group) - 1:
                    delay_seconds = 0.5 # Reduced delay for faster testing
                    logger.info(f"\n...Pausing for {delay_seconds} seconds before next query in this group...\n" + "-"*70 + "\n") # English pause message
                    time.sleep(delay_seconds)

            logger.info(f"\n{'#'*70}\n>>> All Example Queries of Type [{query_group_name}] Processed <<<\n{'#'*70}\n") # English group completion message
            time.sleep(0.5) # Slightly longer pause between groups.

    else:
        # If Retriever or Generator failed to initialize, log a critical error and explain why.
        logger.critical("\nCRITICAL SYSTEM ISSUE: RAG query example flow cannot execute because one or more core components failed to initialize.") # English critical issue
        if not retriever_instance:
            logger.critical("  - Reason: Retriever initialization failed.") # English reason
            logger.critical("    Please carefully review the logs for Step 2 (Indexer initialization) and Step 3 (Retriever initialization) to identify the root cause.") # English instruction
        if not generator_instance:
            logger.critical("  - Reason: Generator initialization failed.") # English reason
            logger.critical("    Please carefully review the logs for Step 4 (Generator initialization), especially regarding the ZHIPUAI_API_KEY check and ZhipuAI client initialization status.") # English instruction
        logger.critical("Please resolve the initialization issues and retry.") # English instruction

    logger.info("--- [Main Flow] Step 5 (RAG Query Examples) Complete. ---\n")

    # =============================================================================================
    # Step 6: Cleanup and Close Resources
    # Mandatory step before program exit to ensure all resources are properly released or saved.
    # =============================================================================================
    logger.info("--- [Main Flow] Step 6: Cleanup and Close System Resources ---")
    # Close component instances sequentially, if they were successfully initialized.
    if retriever_instance:
        try:
            retriever_instance.close()
        except Exception as e_close_retriever:
             logger.error(f"Error during Retriever closing: {e_close_retriever}", exc_info=True)
    else:
        logger.info("  Retriever was not initialized or failed, no cleanup needed.") # English info

    if generator_instance:
        try:
            generator_instance.close()
        except Exception as e_close_generator:
            logger.error(f"Error during Generator closing: {e_close_generator}", exc_info=True)
    else:
        logger.info("  Generator was not initialized or failed, no cleanup needed.") # English info

    if indexer_instance:
        try:
            indexer_instance.close() # Indexer's close method handles saving Faiss indices.
        except Exception as e_close_indexer:
             logger.error(f"Error during Indexer closing (Faiss indices might not be saved): {e_close_indexer}", exc_info=True)
    else:
        logger.info("  Indexer was not initialized or failed, no cleanup needed (Faiss indices may not have been saved).") # English info

    logger.info("--- [Main Flow] System resource cleanup and closing process complete. ---\n")

    logger.info("\n" + "="*80)
    logger.info("========= Multimodal RAG System Example Program Execution Finished =========") # English finish message
    logger.info(f"All output (logs, database, indices, query results) saved to the fixed top-level directory:") # English summary
    logger.info(f"  {os.path.abspath(OUTPUT_BASE_DIR)}")
    logger.info("Key subdirectory overview:") # English overview
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'logs/')}") # Updated path
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'data_storage', 'database/')}") # Updated path
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'data_storage', 'vector_indices/')}") # Updated path
    logger.info(f"  - {os.path.join(OUTPUT_BASE_DIR, 'query_session_results/')}") # Updated path
    logger.info(f"    (Under {os.path.basename(QUERY_RESULTS_DIR)}/, each 'query_XXX_...' subdirectory contains detailed input/output for a single query)") # English explanation
    logger.warning(f"REMINDER: Since the top-level directory '{OUTPUT_BASE_DIR}' is fixed, running the script again with the same identifier will OVERWRITE its contents.") # English reminder
    logger.info("="*80 + "\n")
