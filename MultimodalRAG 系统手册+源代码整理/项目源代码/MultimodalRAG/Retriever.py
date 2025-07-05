from Indexer import Indexer
import logging  # 导入日志模块。这是追踪程序运行状态、诊断问题、记录信息、警告和错误的核心工具。详细的日志是确保系统可维护性的基石。

# -------------------------------------------------------------------------------------------------
# 全局日志记录器设置 (在 `if __name__ == "__main__":` 中会进一步精细配置，这里只是初始化)
# 这是一个重要的工具，我必须确保它随时可用，以便记录系统的每一个动作和潜在问题。
# -------------------------------------------------------------------------------------------------
logger = logging.getLogger(
    __name__
)  # 初始化一个模块级别的日志记录器实例。`__name__` 会被设置成当前模块的名称，便于区分日志来源。
from MultimodalEncoder import (
    MultimodalEncoder,
)  # 导入多模态编码器类。它是处理文本和图像数据的核心工具，负责将这些数据转换为向量表示。
import faiss  # 导入 Faiss 库。这是一个由 Facebook AI Research 开发的、用于高效相似度搜索和聚类的向量库。它将是我们的向量检索引擎。
from typing import (
    List,
    Dict,
    Union,
    Optional,
    Any,
)  # 导入类型提示模块。使用类型提示能让代码更清晰、更易于理解和维护，也能帮助静态分析工具发现潜在错误，这是高质量代码的重要保障。(增加了 Any 类型，以适应某些字典中可能包含的更广泛的数据类型)
import os  # 导入操作系统模块。它提供了与操作系统交互的必要功能，例如处理文件路径、检查文件或目录是否存在、以及创建目录等。这些操作对于管理索引和数据文件至关重要。

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np  # 导入 NumPy 库。它提供了高效的数值计算能力，特别是在处理向量（如CLIP模型生成的特征向量）时不可或缺，能极大地提升性能。


# -------------------------------------------------------------------------------------------------
# 检索器类 (Retriever)
# Retriever 是系统的“搜索大脑”，它根据用户的查询在知识库中找到最相关的文档。
# 它的效率和准确性是提供良好RAG体验的关键。
# -------------------------------------------------------------------------------------------------
class Retriever:
    """
    Retriever 类负责处理用户的查询（可以是文本、图像路径或两者结合的多模态查询），
    并从已建立的索引中检索最相关的文档。其工作流程如下：

    1.  **接收查询**: 用户通过 `retrieve` 方法提交查询。
    2.  **查询编码**: 利用与 `Indexer` 中相同的 `MultimodalEncoder` 实例对用户查询进行向量化，
        将其转换为与索引文档相同向量空间中的特征向量（可能包括文本向量、图像向量和/或平均向量）。保持编码器一致性是保证向量空间对齐的基础。
    3.  **选择策略**: 根据查询的类型（纯文本、纯图像、多模态）和可用性，
        选择最合适的 Faiss 索引（文本索引、图像索引或平均向量索引）以及对应的查询向量进行搜索。
        例如，纯文本查询将使用文本向量在文本索引中搜索。如果首选索引不可用，会尝试回退到其他可用的策略。
    4.  **相似度搜索**: 在选定的 Faiss 索引中执行 Top-K 相似度搜索，找出与查询向量最相似的 K 个向量。
        搜索结果是这些向量的 `internal_id` (与数据库主键对应) 和它们与查询向量的相似度得分。
    5.  **获取元数据**: 使用检索到的 `internal_id` 列表，通过 `Indexer` 的接口从 SQLite 数据库中批量获取
        这些最相关文档的完整元数据（如原始ID、文本内容、图像路径等）。这是将向量结果转化为有用信息的过程。
    6.  **结果组合与返回**: 将获取到的文档元数据与它们各自的相似度得分结合起来，
        并按照相似度得分从高到低（表示最相关）排序，最终返回一个包含这些信息的文档列表。

    Retriever 依赖于一个已经初始化并填充了数据和索引的 `Indexer` 实例。
    它复用 `Indexer` 的编码器以保证查询和文档编码的一致性，并访问 `Indexer` 中的 Faiss 索引和数据库。
    我必须确保它能准确、高效地从 Indexer 获取信息。
    """

    def __init__(self, indexer: Indexer):
        """
        初始化 Retriever 实例。我必须确保它能正确地连接到并使用 Indexer 提供的资源。

        Args:
            indexer (Indexer): 一个已经初始化并包含了数据和索引的 `Indexer` 类的实例。
                               Retriever 的所有操作都依赖于这个 `Indexer` 实例提供的资源。

        Raises:
            ValueError: 如果传入的 `indexer` 不是 `Indexer` 类的有效实例，或者该实例似乎缺少
                        必要的 Faiss 索引属性 (text_index, image_index, mean_index) 或编码器，则抛出此异常。
                        一个没有有效索引源的 Retriever 是无法工作的。
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("开始初始化 Retriever...")

        # 验证传入的 indexer 参数的有效性。Retriever 的工作完全依赖于 Indexer，所以这个检查非常重要。
        if not isinstance(indexer, Indexer):
            msg = "Retriever 初始化错误: 需要一个有效的 Indexer 实例，但收到的不是预期的类型。"
            self.logger.critical(msg)
            raise ValueError(msg)

        # 进一步验证 Indexer 实例是否已成功创建了所需的 Faiss 索引对象和编码器。
        # 缺少这些核心组件意味着 Indexer 初始化失败，Retriever 也无法工作。
        required_indices_attributes = [
            "text_index",
            "image_index",
            "mean_index",
            "encoder",
            "vector_dimension",
        ]
        missing_attrs = [
            attr
            for attr in required_indices_attributes
            if not hasattr(indexer, attr) or getattr(indexer, attr) is None
        ]
        if missing_attrs:
            msg = f"Retriever 初始化错误: 提供的 Indexer 实例缺少以下必需的属性: {', '.join(missing_attrs)}。请确保 Indexer 已成功初始化。"
            self.logger.critical(msg)
            raise ValueError(msg)

        # 保存对传入的 Indexer 实例的引用。Retriever 将通过这个引用访问 Indexer 的资源。
        self.indexer: Indexer = indexer
        # 复用 Indexer 内部的 Encoder 实例。这是确保查询向量和文档向量在同一空间的关键。
        self.encoder: MultimodalEncoder = self.indexer.encoder
        # 从 Indexer 获取向量维度。
        self.vector_dimension: int = self.indexer.vector_dimension
        self.logger.info(
            f"  Retriever 将使用 Indexer 的编码器 (向量维度: {self.vector_dimension})。"
        )

        # 获取对 Indexer 中三个 Faiss 索引的直接引用。
        self.text_index: faiss.Index = self.indexer.text_index
        self.image_index: faiss.Index = self.indexer.image_index
        self.mean_index: faiss.Index = self.indexer.mean_index

        # 检查所有关联的 Faiss 索引是否都为空。如果索引为空，Retriever 就无法找到任何结果。
        text_index_ntotal = getattr(self.text_index, "ntotal", 0)
        image_index_ntotal = getattr(self.image_index, "ntotal", 0)
        mean_index_ntotal = getattr(self.mean_index, "ntotal", 0)

        if (
            text_index_ntotal == 0
            and image_index_ntotal == 0
            and mean_index_ntotal == 0
        ):
            self.logger.warning(
                "Retriever 初始化警告: Indexer 中的所有 Faiss 索引当前都为空。"
            )
            self.logger.warning("  这意味着任何检索操作都将无法找到任何匹配的文档。")
            self.logger.warning(
                "  请确保 Indexer 已成功索引了数据，或者检查索引建立过程的日志。"
            )
        else:
            self.logger.info(f"Retriever 初始化成功。关联的 Indexer 状态如下:")
            self.logger.info(f"    - 文本(Text)索引中向量数: {text_index_ntotal}")
            self.logger.info(f"    - 图像(Image)索引中向量数: {image_index_ntotal}")
            self.logger.info(f"    - 平均(Mean)索引中向量数: {mean_index_ntotal}")
        self.logger.info("Retriever 初始化完成。我已经准备好根据您的查询进行搜索了。")

    def retrieve(
        self, query: Union[str, Dict[str, str]], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        执行完整的检索流程：接收用户查询 -> 对查询进行编码 -> 根据查询类型选择合适的索引和查询向量 ->
        在选定的 Faiss 索引中搜索相似向量 -> 获取这些向量对应的原始文档元数据 -> 组合信息并返回结果。
        这是 Retriever 的核心功能实现。

        Args:
            query (Union[str, Dict[str, str]]): 用户提交的查询。可以是：
                - str: 一个纯文本查询字符串。
                - Dict: 一个字典，用于表示更复杂的查询类型：
                    - {'text': '文本内容', 'image_path': '图像路径'} : 多模态查询，结合文本和图像信息。
                    - {'image_path': '图像路径'} : 纯图像查询，仅使用图像内容进行检索。
                    - {'text': '文本内容'} : 纯文本查询 (与直接传入字符串的效果相同，但通过字典形式提供)。
                    字典中必须至少包含 'text' 或 'image_path' 键及其对应值，且值不能为空字符串。
            k (int): 指定希望检索的最相似文档的数量 (Top-K)。默认为 5。

        Returns:
            List[Dict[str, Any]]: 一个按相似度得分降序排列的文档列表。
                        列表中的每个字典代表一个检索到的文档，包含以下键（但不限于）：
                        - 'id': 原始文档 ID (str)。
                        - 'text': 文档的文本内容 (str 或 None)。
                        - 'image_path': 关联图像的路径 (str 或 None)。
                        - 'internal_id': 数据库和 Faiss 使用的内部 ID (int)。
                        - 'score': 该文档与查询的相似度得分 (float)。对于内积搜索，得分越高表示越相似。
                        如果查询无效、编码失败、所选索引为空或搜索无结果，则返回空列表 `[]`。
        """
        self.logger.info(f"开始执行检索流程，目标是获取 Top-{k} 最相关的文档...")
        # 为了日志清晰，截断长查询字符串。
        query_str_for_log = str(query)
        self.logger.debug(
            f"  接收到的原始查询: {query_str_for_log[:200]}{'...' if len(query_str_for_log)>200 else ''}, k={k}"
        )

        query_text: Optional[str] = None
        query_image_path: Optional[str] = None
        query_type: str = "unknown"

        # --- 步骤 1: 解析查询输入，确定查询类型和具体内容 ---
        self.logger.debug("  - Retriever步骤 1: 解析用户查询输入...")
        if isinstance(query, str):
            query_text_stripped = query.strip()
            if query_text_stripped:
                query_text = query_text_stripped
                query_type = "纯文本"
                self.logger.info(f"    查询类型确定为: {query_type} (字符串输入)")
                self.logger.info(
                    f"    查询文本内容: '{query_text[:100]}{'...' if len(query_text)>100 else ''}'"
                )
            else:
                self.logger.error("查询错误: 纯文本查询字符串为空或只包含空白。")
                return []
        elif isinstance(query, dict):
            # 从字典中安全地获取文本和图像路径。
            query_text_from_dict = query.get("text")
            query_image_path_from_dict = query.get("image_path")

            # 确保获取到的值是字符串，且去除空白后非空。
            query_text = (
                query_text_from_dict.strip()
                if isinstance(query_text_from_dict, str)
                and query_text_from_dict.strip()
                else None
            )
            query_image_path = (
                query_image_path_from_dict.strip()
                if isinstance(query_image_path_from_dict, str)
                and query_image_path_from_dict.strip()
                else None
            )

            # 根据有效的输入组合确定查询类型。
            if query_text and query_image_path:
                query_type = "多模态"
                self.logger.info(f"    查询类型确定为: {query_type}")
                self.logger.info(
                    f"    查询文本部分: '{query_text[:50]}{'...' if len(query_text)>50 else ''}'"
                )
                self.logger.info(
                    f"    查询图像部分: '{os.path.basename(query_image_path)}'"
                )
            elif query_image_path:
                # 检查图像文件是否存在
                if os.path.exists(query_image_path) and os.path.isfile(
                    query_image_path
                ):
                    query_type = "纯图像"
                    self.logger.info(f"    查询类型确定为: {query_type}")
                    self.logger.info(
                        f"    查询图像路径: '{os.path.basename(query_image_path)}' (文件存在)"
                    )
                else:
                    self.logger.error(
                        f"查询错误: 纯图像查询指定的图像文件路径无效或不存在: '{query_image_path}'"
                    )
                    return []
            elif query_text:
                query_type = "纯文本"
                self.logger.info(f"    查询类型确定为: {query_type} (字典输入)")
                self.logger.info(
                    f"    查询文本内容: '{query_text[:100]}{'...' if len(query_text)>100 else ''}'"
                )
            else:
                self.logger.error(
                    "查询错误: 查询字典无效，必须至少包含有效的非空 'text' 或有效的非空 'image_path' 键及其对应值。"
                )
                return []
        else:
            self.logger.error(
                f"查询错误: 不支持的查询类型 ({type(query)}) 或查询内容为空。查询必须是有效的非空字符串或包含有效内容的字典。"
            )
            return []

        # --- 步骤 2: 使用内部的 MultimodalEncoder 对查询进行编码 ---
        self.logger.debug(
            f"  - Retriever步骤 2: 使用 MultimodalEncoder 对 '{query_type}' 查询进行编码..."
        )
        encoded_query_vectors: Dict[str, Optional[np.ndarray]]
        try:
            # 调用 Indexer 的编码器对查询进行编码。
            encoded_query_vectors = self.encoder.encode(
                text=query_text, image_path=query_image_path
            )
            query_text_vec = encoded_query_vectors.get("text_vector")
            query_image_vec = encoded_query_vectors.get("image_vector")
            query_mean_vec = encoded_query_vectors.get("mean_vector")

            # 如果编码器未能生成任何向量，则无法进行检索。
            if (
                query_text_vec is None
                and query_image_vec is None
                and query_mean_vec is None
            ):
                self.logger.warning(
                    "查询编码警告: MultimodalEncoder 未能为当前查询生成任何有效的特征向量。无法继续检索。"
                )
                return []
            self.logger.info("    查询编码完成。")
            if query_text_vec is not None:
                self.logger.debug("      - 生成了文本查询向量。")
            if query_image_vec is not None:
                self.logger.debug("      - 生成了图像查询向量。")
            if query_mean_vec is not None:
                self.logger.debug("      - 生成了平均查询向量。")

        except Exception as e:
            self.logger.error(
                f"查询编码严重错误: 在对查询进行编码时发生意外错误: {e}", exc_info=True
            )
            return []

        # --- 步骤 3: 根据查询类型选择目标 Faiss 索引和相应的查询向量 ---
        # 这是决定使用哪个索引进行搜索的逻辑。优先级通常是 多模态 -> 文本 -> 图像。
        self.logger.debug(
            f"  - Retriever步骤 3: 根据查询类型 '{query_type}' 选择搜索策略 (Faiss索引和查询向量)..."
        )
        target_faiss_index: Optional[faiss.Index] = None  # 最终选定的 Faiss 索引对象。
        search_query_vector: Optional[np.ndarray] = None  # 最终用于搜索的查询向量。
        selected_index_name: str = "N/A"  # 用于日志记录的索引名称。

        # 获取当前索引中的向量数量，用于判断索引是否可用。
        text_index_ntotal = getattr(self.text_index, "ntotal", 0)
        image_index_ntotal = getattr(self.image_index, "ntotal", 0)
        mean_index_ntotal = getattr(self.mean_index, "ntotal", 0)

        if query_type == "纯文本":
            # 纯文本查询优先使用文本向量和文本索引。
            if query_text_vec is not None and text_index_ntotal > 0:
                target_faiss_index = self.text_index
                search_query_vector = query_text_vec
                selected_index_name = "文本(Text)索引"
                self.logger.info(
                    f"    搜索策略: 使用文本查询向量，在 {selected_index_name} (含 {text_index_ntotal} 个向量) 中搜索。"
                )
            else:
                reason = (
                    "文本查询向量编码失败"
                    if query_text_vec is None
                    else f"文本(Text) Faiss 索引为空 (仅含 {text_index_ntotal} 个向量)"
                )
                self.logger.warning(f"无法执行纯文本查询，因为: {reason}。")
                return []
        elif query_type == "纯图像":
            # 纯图像查询优先使用图像向量和图像索引。
            if query_image_vec is not None and image_index_ntotal > 0:
                target_faiss_index = self.image_index
                search_query_vector = query_image_vec
                selected_index_name = "图像(Image)索引"
                self.logger.info(
                    f"    搜索策略: 使用图像查询向量，在 {selected_index_name} (含 {image_index_ntotal} 个向量) 中搜索。"
                )
            else:
                reason = (
                    "图像查询向量编码失败"
                    if query_image_vec is None
                    else f"图像(Image) Faiss 索引为空 (仅含 {image_index_ntotal} 个向量)"
                )
                self.logger.warning(f"无法执行纯图像查询，因为: {reason}。")
                return []
        elif query_type == "多模态":
            # 多模态查询优先使用平均向量和平均索引。
            if query_mean_vec is not None and mean_index_ntotal > 0:
                target_faiss_index = self.mean_index
                search_query_vector = query_mean_vec
                selected_index_name = "平均(Mean)索引"
                self.logger.info(
                    f"    搜索策略: 使用平均查询向量，在 {selected_index_name} (含 {mean_index_ntotal} 个向量) 中搜索。"
                )
            # 如果平均向量或平均索引不可用，尝试回退到使用文本向量在文本索引中搜索。
            elif query_text_vec is not None and text_index_ntotal > 0:
                self.logger.warning(
                    "多模态查询警告: 平均(Mean)索引或平均查询向量不可用/索引为空。"
                )
                self.logger.info(
                    f"    应用回退策略: 改为使用文本查询向量，在文本(Text)索引 (含 {text_index_ntotal} 个向量) 中搜索。"
                )
                target_faiss_index = self.text_index
                search_query_vector = query_text_vec
                selected_index_name = "文本(Text)索引 (作为多模态查询的回退)"
            # 如果所有首选和回退策略都不可用。
            else:
                reason_parts = []
                if query_mean_vec is None:
                    reason_parts.append("平均查询向量编码失败")
                if query_text_vec is None:
                    reason_parts.append("文本查询向量编码失败")
                if mean_index_ntotal == 0:
                    reason_parts.append(
                        f"平均(Mean)索引为空 (含{mean_index_ntotal}向量)"
                    )
                if text_index_ntotal == 0:
                    reason_parts.append(
                        f"文本(Text)索引为空 (含{text_index_ntotal}向量)"
                    )

                final_reason = (
                    "; ".join(reason_parts)
                    if reason_parts
                    else "由于所有可能的搜索策略（平均、文本回退）都不可用或对应的查询向量缺失"
                )

                self.logger.warning(f"无法执行多模态查询，因为: {final_reason}。")
                return []
        else:
            # 如果查询类型是未知的，这是一个内部逻辑问题。
            self.logger.error(
                "内部逻辑错误: 无法为当前查询确定有效的查询类型或找不到可用的查询向量/索引组合。无法继续搜索。"
            )
            return []

        # 最终确认是否成功选择了搜索目标。
        if target_faiss_index is None or search_query_vector is None:
            self.logger.error(
                "内部错误: 搜索目标 Faiss 索引或查询向量未能正确设置，尽管已尝试选择策略。无法继续搜索。"
            )
            return []

        # --- 步骤 4: 在选定的 Faiss 索引中执行 Top-K 相似度搜索 ---
        # 这是使用 Faiss 核心功能的步骤。
        self.logger.debug(
            f"  - Retriever步骤 4: 在选定的 '{selected_index_name}' 中执行 Faiss Top-{k} 搜索..."
        )
        try:
            # Faiss search 方法期望输入的是一个二维数组 (batch_size, vector_dimension)。
            # 我们的查询向量是单个向量，所以需要将其 reshape 成 (1, vector_dimension)。
            # 确保向量是 float32 类型，Faiss 通常需要这个类型。
            query_vector_for_faiss = search_query_vector.astype("float32").reshape(
                1, self.vector_dimension
            )

            self.logger.debug(
                f"    Faiss search: k={k}, query_vector_shape={query_vector_for_faiss.shape}"
            )
            # 执行 Faiss 搜索。它返回距离/得分矩阵和对应的 ID 矩阵。
            scores_matrix, internal_ids_matrix = target_faiss_index.search(
                query_vector_for_faiss, k
            )
            self.logger.debug(
                f"    Faiss search returned scores_matrix shape: {scores_matrix.shape}, ids_matrix shape: {internal_ids_matrix.shape}"
            )

            retrieved_internal_ids: List[int] = []
            retrieved_scores: List[float] = []

            # 遍历搜索结果。Faiss 返回的 ID 矩阵可能包含 -1，表示未找到足够多的结果或填充。
            # 我们只收集有效的 ID (不等于 -1)。
            if (
                internal_ids_matrix.size > 0 and scores_matrix.size > 0
            ):  # Check if results are not empty
                for id_val, score_val in zip(internal_ids_matrix[0], scores_matrix[0]):
                    if id_val != -1:
                        retrieved_internal_ids.append(
                            int(id_val)
                        )  # 确保 ID 是整数类型。
                        retrieved_scores.append(
                            float(score_val)
                        )  # 确保得分是浮点数类型。
                    else:
                        # Faiss often pads with -1 when fewer than k results are found.
                        self.logger.debug(
                            f"    Faiss search: Encountered -1 ID, indicating fewer than k={k} results or padding. Stopping collection for this query."
                        )
                        break  # Stop collecting if -1 is encountered
            else:
                self.logger.debug(
                    "    Faiss search: Returned empty ID or score matrices."
                )

            if not retrieved_internal_ids:
                self.logger.info(
                    f"    Faiss 搜索在 '{selected_index_name}' 中完成，但未返回任何有效的结果 ID。可能该索引为空或没有相似度足够高的向量。"
                )
                return []

            self.logger.info(
                f"    Faiss 搜索在 '{selected_index_name}' 中完成，初步找到 {len(retrieved_internal_ids)} 个候选文档的 internal_id。"
            )

        except Exception as e:
            self.logger.error(
                f"Faiss 搜索错误: 在 '{selected_index_name}' 中执行 Faiss 搜索时发生错误: {e}",
                exc_info=True,
            )
            return []

        # --- 步骤 5: 根据检索到的 internal_ids 从 SQLite 数据库批量获取这些文档的完整元数据 ---
        # 有了 internal_id，我们需要从数据库获取原始的文本、图像路径等信息。
        self.logger.debug(
            f"  - Retriever步骤 5: 使用找到的 {len(retrieved_internal_ids)} 个 internal_id，从 SQLite 数据库批量获取文档元数据..."
        )
        documents_map_from_db: Dict[int, Dict[str, Any]]
        if retrieved_internal_ids:
            documents_map_from_db = self.indexer.get_documents_by_internal_ids(
                retrieved_internal_ids
            )
            self.logger.info(
                f"    已成功从数据库中获取了 {len(documents_map_from_db)} 条与 internal_id 对应的文档记录。"
            )
        else:
            # 如果 Faiss 没有返回有效 ID，则无需查询数据库。
            self.logger.info("    由于 Faiss 未返回有效 ID，跳过数据库元数据获取步骤。")
            documents_map_from_db = {}

        # --- 步骤 6: 组合结果：将元数据与相似度得分结合，并保持 Faiss 返回的原始排序 ---
        # Faiss 返回的结果已经是按相似度排序的，我们只需将元数据和得分结合起来。
        self.logger.debug(
            f"  - Retriever步骤 6: 组合文档元数据与相似度得分，并按 Faiss 原始顺序排列..."
        )
        final_retrieved_docs: List[Dict[str, Any]] = []

        # 遍历 Faiss 返回的 internal_id 和 score 列表（它们是同步排序的）。
        for internal_id, score in zip(retrieved_internal_ids, retrieved_scores):
            # 从数据库查询结果字典中查找对应的文档元数据。
            doc_data_from_db = documents_map_from_db.get(internal_id)

            if doc_data_from_db:
                # 如果找到了元数据，将相似度得分添加到文档字典中。
                doc_data_from_db["score"] = score
                final_retrieved_docs.append(doc_data_from_db)
            else:
                # 如果 Faiss 返回的 ID 在数据库中找不到，说明存在数据不一致。这是一个警告，需要记录。
                self.logger.warning(
                    f"数据不一致警告: 在数据库中未能找到 Faiss 返回的 internal_id: {internal_id}。"
                )
                self.logger.warning(
                    f"                 这可能表示 Faiss 索引与数据库元数据之间存在不一致。将跳过此条检索结果。"
                )

        self.logger.info(
            f"检索流程成功完成，最终返回 {len(final_retrieved_docs)} 个文档（已按相似度排序）。"
        )
        return final_retrieved_docs

    def close(self):
        """
        关闭 Retriever 实例时调用的清理方法。
        Retriever 本身通常没有需要显式关闭的外部资源 (因为它主要依赖于 Indexer 提供的资源)。
        此方法主要用于记录 Retriever 的关闭事件。这是负责任的结束流程的一部分。
        """
        self.logger.info("开始关闭 Retriever 实例...")
        # 通常无需执行特定的资源释放操作，因为 Encoder, Faiss索引, DB连接等由 Indexer 管理。
        self.logger.info("Retriever 实例关闭完成。")
