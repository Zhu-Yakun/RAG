import logging  # 导入日志模块。这是追踪程序运行状态、诊断问题、记录信息、警告和错误的核心工具。详细的日志是确保系统可维护性的基石。
from transformers import (
    CLIPProcessor,
    CLIPModel,
)  # 从 Hugging Face Transformers 库导入 CLIP 模型的处理器和模型本身。

# CLIP (Contrastive Language–Image Pre-training) 是一个强大的多模态模型，能够将文本和图像编码到同一个向量空间，这是我们实现多模态检索的关键技术。
# 安装提示: pip install transformers torch pillow。
import torch  # 导入 PyTorch 库。这是一个广泛使用的开源机器学习框架，Transformers 库基于它构建。我们使用它来加载和运行CLIP模型，并利用其GPU加速能力（如果可用）。
import numpy as np  # 导入 NumPy 库。它提供了高效的数值计算能力，特别是在处理向量（如CLIP模型生成的特征向量）时不可或缺，能极大地提升性能。
from typing import (
    Dict,
    Optional,
)  # 导入类型提示模块。使用类型提示能让代码更清晰、更易于理解和维护，也能帮助静态分析工具发现潜在错误，这是高质量代码的重要保障。(增加了 Any 类型，以适应某些字典中可能包含的更广泛的数据类型)
from PIL import (
    Image,
    UnidentifiedImageError,
)  # 导入 Pillow 库 (PIL 的一个分支)。它是Python中事实上的图像处理标准库，用于图像文件的加载、处理和保存。 (增加了 UnidentifiedImageError 用于捕获特定的图像加载错误)
import os  # 导入操作系统模块。它提供了与操作系统交互的必要功能，例如处理文件路径、检查文件或目录是否存在、以及创建目录等。这些操作对于管理索引和数据文件至关重要。

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------------------------------------------------------------------------
# 全局日志记录器设置 (在 `if __name__ == "__main__":` 中会进一步精细配置，这里只是初始化)
# 这是一个重要的工具，我必须确保它随时可用，以便记录系统的每一个动作和潜在问题。
# -------------------------------------------------------------------------------------------------
logger = logging.getLogger(
    __name__
)  # 初始化一个模块级别的日志记录器实例。`__name__` 会被设置成当前模块的名称，便于区分日志来源。


# -------------------------------------------------------------------------------------------------
# 多模态编码器类 (MultimodalEncoder)
# 这是一个核心组件，负责将原始数据转化为机器可理解的向量表示。它的准确性直接影响检索效果。
# -------------------------------------------------------------------------------------------------
class MultimodalEncoder:
    """
    使用 Hugging Face Transformers 库中的 CLIP (Contrastive Language–Image Pre-training) 模型
    来对文本和/或图像进行编码，将它们转换为高维向量表示 (特征向量)。
    CLIP 模型能够将文本和图像映射到同一个语义向量空间，使得它们的向量表示具有可比性，
    这是多模态检索和理解的基础。

    核心功能:
    - 在初始化时加载预训练的 CLIP 模型和对应的处理器 (processor)。
    - 提供 `encode` 方法，该方法可以接受文本字符串、图像文件路径，或两者都接受。
    - `encode` 方法对输入进行预处理、通过 CLIP 模型进行编码，然后对输出的向量进行 L2 归一化。
    - 返回一个字典，包含文本向量、图像向量以及（如果两者都提供了）两者的平均向量。
    - 自动检测并优先使用 GPU (如果 CUDA 可用) 进行计算加速，否则回退到 CPU。
    - L2 归一化对于后续使用 Faiss 进行基于内积 (Inner Product) 的相似度搜索至关重要，
      因为归一化向量的内积等价于它们之间的余弦相似度。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化 MultimodalEncoder 类。我需要确保模型和处理器能够被正确加载，这是编码器能工作的先决条件。

        Args:
            model_name (str): 指定要加载的 Hugging Face Hub 上的 CLIP 模型名称。
                              例如 "openai/clip-vit-base-patch32"。
                              不同的 CLIP 模型变体具有不同的性能、速度和输出向量维度。
                              选择合适的模型取决于具体的应用需求和可用资源。
                              请注意: "openai/clip-vit-base-patch32" 是一个性能和资源消耗均衡的基准模型。
                              若资源极度受限，可研究更轻量模型，但可能影响编码质量。

        Raises:
            Exception: 如果在加载 CLIP 模型或处理器时发生任何错误（例如，网络问题导致无法下载模型文件、
                       指定的模型名称无效、或者相关的依赖库未正确安装），则会抛出异常。
                       由于模型是编码器的核心，加载失败意味着编码器无法工作，这需要立即报告为严重错误。
        """
        # 获取一个特定于此类实例的日志记录器，方便追踪和调试。
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(
            f"开始初始化 MultimodalEncoder，尝试加载 CLIP 模型: {model_name}"
        )

        try:
            # 步骤 1: 加载与指定 CLIP 模型相关联的处理器 (CLIPProcessor)。
            # 处理器负责将原始的文本和图像数据转换为 CLIP 模型期望的输入格式。
            # 对于文本，这通常包括分词 (tokenization)、添加特殊标记、转换为 token ID。
            # 对于图像，这通常包括调整大小 (resizing)、归一化 (normalization) 像素值。
            # 这是数据进入模型的必经之路，不能出错。
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.logger.info(f"CLIP Processor for '{model_name}' 加载成功。")

            # 步骤 2: 加载预训练的 CLIP 模型本身 (CLIPModel)。模型是编码功能的核心。
            self.model = CLIPModel.from_pretrained(model_name)
            self.logger.info(f"CLIP Model '{model_name}' 加载成功。")

            # 步骤 3: 获取模型的输出向量维度。这个维度信息对于构建 Faiss 索引是必需的。
            # 对于 CLIP 模型，文本编码器和图像编码器的输出向量维度通常是相同的。
            # text_model.config.hidden_size 通常存储了这个维度值。
            self.vector_dimension = self.model.text_model.config.hidden_size
            self.logger.info(f"CLIP 模型的特征向量维度为: {self.vector_dimension}")

            # 步骤 4: 将模型设置为评估模式 (evaluation mode)。
            # 调用 .eval() 会关闭模型中的 Dropout 层和 Batch Normalization 层的更新行为。
            # 这对于推理（编码）阶段非常重要，以确保结果的一致性和确定性。这是标准做法。
            self.model.eval()

            # 步骤 5: 检测可用的计算设备 (GPU 或 CPU)，并将模型迁移到该设备。
            # 优先使用 GPU 以提高编码速度，这对于处理大量数据时非常重要。
            if torch.cuda.is_available():  # 检查系统中是否有可用的 CUDA GPU。
                self.device = torch.device("cuda")  # 如果有，则选择使用 GPU。
                self.logger.info(
                    "检测到 CUDA 支持，模型将运行在 GPU 上以获得更快的编码速度。"
                )
            else:
                self.device = torch.device("cpu")  # 如果没有 GPU，则使用 CPU。
                self.logger.info(
                    "未检测到 CUDA 支持，模型将运行在 CPU 上 (编码速度可能较慢)。"
                )

            self.model.to(self.device)  # 将模型的所有参数和缓冲区移动到选定的设备。
            self.logger.info(f"模型已成功移动到设备: {self.device}")
            self.logger.info(
                "MultimodalEncoder 初始化成功完成。我已经准备好进行编码工作了。"
            )

        except Exception as e:
            # 如果在上述任何步骤中发生错误，记录详细的错误信息并重新抛出异常。加载模型的失败是致命的，必须报告。
            self.logger.critical(
                f"初始化 MultimodalEncoder 失败：加载 CLIP 模型 '{model_name}' 时发生严重错误。"
            )
            self.logger.error(
                f"错误详情: {e}", exc_info=True
            )  # exc_info=True 会记录完整的堆栈跟踪，便于诊断。
            self.logger.error("请检查以下几点：")
            self.logger.error(
                f"  1. 确保指定的模型名称 '{model_name}' 正确且在 Hugging Face Hub 上可用。"
            )
            self.logger.error(
                "  2. 确保已正确安装必要的 Python 库: 'transformers', 'torch', 'pillow'。"
            )
            self.logger.error(
                "     (例如，通过命令: pip install transformers torch pillow)"
            )
            self.logger.error(
                "  3. 确保网络连接正常，以便能够从 Hugging Face Hub 下载模型文件 (首次加载时需要)。"
            )
            raise RuntimeError(
                f"MultimodalEncoder 初始化失败: {e}"
            ) from e  # 重新抛出异常，表明初始化失败。

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        对输入的 NumPy 向量进行 L2 范数归一化 (L2 Normalization)。
        L2 归一化将向量缩放，使其 L2 范数（欧几里得长度）为 1。
        这对于计算余弦相似度非常重要：两个 L2 归一化向量的点积（内积）等于它们之间的余弦相似度。
        保持向量归一化是使用内积进行相似度搜索的必要前处理。

        Args:
            vector (np.ndarray): 需要进行 L2 归一化的 NumPy 浮点数向量。

        Returns:
            np.ndarray: 经过 L2 归一化后的向量。如果输入向量的范数非常接近于零（即零向量），
                        则直接返回一个相同形状的零向量，以避免除以零的错误。这是为了处理特殊情况，确保程序的健壮性。
        """
        # 计算向量的 L2 范数 (向量的欧几里得长度)。
        norm = np.linalg.norm(vector)

        # 检查范数是否大于一个很小的阈值 (epsilon)，以避免除以零或因浮点数精度问题导致的数值不稳定。
        # 1e-9 是一个常用的小正数，用于判断一个浮点数是否“接近于零”。
        if norm > 1e-9:
            # 如果范数足够大，则将向量的每个元素除以该范数，得到归一化向量。
            return vector / norm
        else:
            # 如果范数非常小（向量接近零向量），直接返回一个与输入向量形状相同但所有元素为零的向量。
            # 这是对数值稳定性的考虑。
            self.logger.debug("尝试归一化一个范数接近零的向量。返回零向量。")
            return np.zeros_like(vector)

    def encode(
        self, text: Optional[str] = None, image_path: Optional[str] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        对输入的文本字符串和/或图像文件路径进行编码，生成它们对应的归一化特征向量。
        这是将原始数据转化为向量的核心操作。

        Args:
            text (Optional[str]): 需要编码的文本字符串。如果为 None 或空字符串，则不进行文本编码。
            image_path (Optional[str]): 需要编码的图像文件的完整路径。如果为 None 或路径无效，则不进行图像编码。

        Returns:
            Dict[str, Optional[np.ndarray]]: 一个字典，包含以下可能的键值对：
                - 'text_vector': 如果提供了有效的文本且编码成功，则为该文本的 L2 归一化 NumPy 向量 (float32)。否则为 None。
                - 'image_vector': 如果提供了有效的图像路径、图像文件可读且编码成功，则为该图像的 L2 归一化 NumPy 向量 (float32)。否则为 None。
                - 'mean_vector': 如果文本和图像都提供了，并且两者都成功编码，则为两者特征向量的 L2 归一化平均向量 (float32)。
                                 这个平均向量可以作为文本和图像结合的多模态表示。如果任一编码失败或未提供，则为 None。
            如果 `text` 和 `image_path` 都为 None，将记录错误并返回所有值为 None 的字典。
        """
        # 输入有效性检查：必须至少提供文本或图像路径之一。如果什么都没提供，就无法编码。
        is_text_valid = text is not None and text.strip()
        is_image_path_valid = image_path is not None and image_path.strip()

        if not is_text_valid and not is_image_path_valid:
            self.logger.error(
                "编码错误：必须至少提供有效的非空文本或有效的图像路径才能进行编码。"
            )
            return {"text_vector": None, "image_vector": None, "mean_vector": None}

        # 初始化各个向量为 None，它们将在编码成功后被赋值。
        text_vector: Optional[np.ndarray] = None
        image_vector: Optional[np.ndarray] = None
        mean_vector: Optional[np.ndarray] = None

        # 使用 torch.no_grad() 上下文管理器进行推理。
        # 这会禁用 PyTorch 的梯度计算，从而减少内存消耗并加速计算，因为在编码（推理）阶段不需要进行反向传播。这是提高效率的标准做法。
        with torch.no_grad():
            # --- 步骤 A: 编码文本 (如果提供了文本) ---
            if is_text_valid:  # 确保文本非None且非空（去除两端空白后）
                self.logger.debug(
                    f"开始编码文本: '{text[:50]}{'...' if len(text)>50 else ''}'"
                )
                try:
                    # 1. 预处理文本: 使用 CLIP Processor 将文本字符串转换为模型所需的输入格式。
                    #    `return_tensors="pt"`: 返回 PyTorch 张量 (tensors)。
                    #    `padding=True`: 将批次内的文本填充到相同长度 (批次中最长文本的长度)。
                    #    `truncation=True`: 如果文本超过模型的最大输入长度，则进行截断。
                    #    `.to(self.device)`: 将生成的输入张量移动到之前确定的计算设备 (CPU 或 GPU)。
                    text_inputs = self.processor(
                        text=text, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)

                    # 2. 获取文本特征: 调用 CLIP 模型的 `get_text_features` 方法，传入预处理后的输入。
                    #    使用 `**text_inputs` 将字典解包为关键字参数。
                    text_features_tensor = self.model.get_text_features(**text_inputs)

                    # 3. 后处理特征张量:
                    #    `.squeeze()`: 如果批次大小为1，移除批次维度，得到一个一维张量 (向量)。
                    #    `.cpu()`: 将结果张量从 GPU (如果在使用) 移回 CPU，因为 NumPy 操作通常在 CPU 上进行。
                    #    `.numpy()`: 将 PyTorch 张量转换为 NumPy 数组。
                    #    `.astype('float32')`: 确保数据类型为 float32，这是 Faiss 常用的数值类型，也节省内存。
                    text_vector_raw = (
                        text_features_tensor.squeeze().cpu().numpy().astype("float32")
                    )

                    # 4. L2 归一化: 对原始的文本特征向量进行 L2 范数归一化。
                    text_vector = self._normalize_vector(text_vector_raw)
                    self.logger.debug("文本编码成功并已归一化。")

                except Exception as e:
                    # 如果文本编码过程中发生任何错误，记录错误信息。这可能是由于模型输入处理问题。
                    self.logger.error(
                        f"编码文本时发生错误。文本: '{text[:50]}...'. 错误详情: {e}",
                        exc_info=False,
                    )  # exc_info=False 避免在每次文本编码失败时都打印完整堆栈
                    text_vector = None  # 确保在失败时 text_vector 为 None。

            # --- 步骤 B: 编码图像 (如果提供了图像路径) ---
            if is_image_path_valid:  # 确保图像路径非None且非空
                self.logger.debug(f"开始编码图像: '{image_path}'")
                try:
                    # 1. 加载图像: 使用 Pillow (PIL) 库的 Image.open() 方法打开图像文件。
                    #    `.convert("RGB")`: 确保图像转换为 RGB 格式。CLIP 模型通常期望 RGB 图像作为输入。
                    #                       即使原始图像是 RGBA 或灰度图，也会被转换为 RGB。这是模型输入的要求。
                    image_pil = Image.open(image_path).convert("RGB")

                    # 2. 预处理图像: 使用 CLIP Processor 将 PIL.Image 对象转换为模型所需的输入格式。
                    #    `return_tensors="pt"`: 返回 PyTorch 张量。
                    #    `.to(self.device)`: 将输入张量移动到计算设备。
                    image_inputs = self.processor(
                        images=image_pil, return_tensors="pt"
                    ).to(self.device)

                    # 3. 获取图像特征: 调用 CLIP 模型的 `get_image_features` 方法。
                    image_features_tensor = self.model.get_image_features(
                        **image_inputs
                    )

                    # 4. 后处理特征张量 (与文本编码类似): 转换为归一化的 NumPy float32 数组。
                    image_vector_raw = (
                        image_features_tensor.squeeze().cpu().numpy().astype("float32")
                    )
                    image_vector = self._normalize_vector(image_vector_raw)
                    self.logger.debug(
                        f"图像 '{os.path.basename(image_path)}' 编码成功并已归一化。"
                    )

                except FileNotFoundError:
                    # 如果指定的图像文件路径不存在。这是文件系统问题。
                    self.logger.warning(
                        f"图像编码警告: 图像文件未找到于路径 '{image_path}'。将跳过此图像的编码。"
                    )
                    image_vector = None
                except UnidentifiedImageError:  # Pillow 无法识别图像格式
                    self.logger.error(
                        f"图像编码错误: 无法识别或打开图像文件 '{image_path}'。文件可能已损坏或格式不受支持。"
                    )
                    image_vector = None
                except Exception as e:
                    # 如果在加载或处理图像时发生其他错误 (例如，图像文件损坏、权限问题)。
                    self.logger.error(
                        f"编码图像 '{image_path}' 时发生错误。错误详情: {e}",
                        exc_info=False,
                    )
                    image_vector = None

        # --- 步骤 C: 计算平均向量 (仅当文本和图像都成功编码时) ---
        # 检查 text_vector 和 image_vector 是否都成功生成 (即它们都不是 None)。只有同时有文本和图像时，计算平均向量才有意义。
        if text_vector is not None and image_vector is not None:
            self.logger.debug("文本和图像均成功编码，开始计算它们的平均向量...")
            try:
                # 1. 计算平均: 使用 NumPy 的 `mean` 函数计算两个向量的逐元素平均值。
                #    `axis=0` 表示沿着第一个轴（即向量本身）计算平均值。
                #    确保结果的数据类型为 float32。
                mean_vector_raw = np.mean(
                    np.array([text_vector, image_vector]), axis=0
                ).astype("float32")

                # 2. L2 归一化: 对计算出的原始平均向量再次进行 L2 归一化。
                #    这很重要，因为两个单位向量的平均向量长度通常不为 1。
                mean_vector = self._normalize_vector(mean_vector_raw)
                self.logger.debug("平均向量计算并归一化成功。")
            except Exception as e:
                # 如果计算平均向量时出错。
                self.logger.error(
                    f"计算文本和图像的平均向量时发生错误。错误详情: {e}", exc_info=False
                )
                mean_vector = None
        elif text_vector is not None or image_vector is not None:
            # 如果只编码了文本或图像之一，则不需要计算平均向量。
            self.logger.debug("仅文本或图像之一被成功编码，因此不计算平均向量。")

        # 总结编码结果，用于日志记录。这提供了每条编码操作的清晰反馈。
        results_summary = []
        if text_vector is not None:
            results_summary.append("文本向量")
        if image_vector is not None:
            results_summary.append("图像向量")
        if mean_vector is not None:
            results_summary.append("平均向量")

        input_summary_parts = []
        if is_text_valid:
            input_summary_parts.append(
                f"文本='{text[:30]}{'...' if len(text)>30 else ''}'"
            )
        if is_image_path_valid:
            input_summary_parts.append(f"图像='{os.path.basename(image_path)}'")
        input_desc = (
            ", ".join(input_summary_parts) if input_summary_parts else "无有效输入"
        )

        # 根据编码结果是否有向量生成，记录不同级别的日志。
        if not results_summary and (is_text_valid or is_image_path_valid):
            self.logger.warning(
                f"编码完成，但对于输入 ({input_desc})，未能生成任何有效向量。"
            )
        elif results_summary:
            self.logger.info(
                f"编码完成对于 ({input_desc})。成功生成的向量: {', '.join(results_summary)}。"
            )

        # 返回包含所有结果向量的字典。
        return {
            "text_vector": text_vector,
            "image_vector": image_vector,
            "mean_vector": mean_vector,
        }
