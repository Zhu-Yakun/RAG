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
)  # 导入自定义的多模态编码器类。它是将文本和图像转换为向量的核心组件。
import os  # 导入操作系统模块。它提供了与操作系统交互的必要功能，例如处理文件路径、检查文件或目录是否存在、以及创建目录等。这些操作对于管理索引和数据文件至关重要。

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sqlite3  # 导入 SQLite 数据库模块。它是我们存储和管理文档元数据的关键，比如文档的内部ID、原始ID、文本描述和图像路径。
import faiss  # 导入 Faiss 库。这是一个由 Facebook AI Research 开发的、用于高效相似度搜索和聚类的向量库。它将是我们的向量检索引擎。

# 安装提示: pip install faiss-cpu (如果您使用CPU) 或 pip install faiss-gpu (如果您有CUDA环境的GPU)。
from typing import (
    List,
    Dict,
    Optional,
    Any,
)  # 导入类型提示模块。使用类型提示能让代码更清晰、更易于理解和维护，也能帮助静态分析工具发现潜在错误，这是高质量代码的重要保障。(增加了 Any 类型，以适应某些字典中可能包含的更广泛的数据类型)
import numpy as np  # 导入 NumPy 库。它提供了高效的数值计算能力，特别是在处理向量（如CLIP模型生成的特征向量）时不可或缺，能极大地提升性能。


# -------------------------------------------------------------------------------------------------
# 索引器类 (Indexer)
# Indexer 是整个系统的知识库构建者和管理者。它的稳定性和数据一致性至关重要。
# 我必须确保数据库和向量索引的正确初始化、填充和持久化。
# -------------------------------------------------------------------------------------------------
class Indexer:
    """
    Indexer 类是多模态 RAG (Retrieval Augmented Generation) 系统的数据管理核心。它负责：
    1.  **接收文档数据**: 从外部（例如，`load_data_from_json_and_associate_images` 函数）获取包含文本和图像路径的文档列表。
    2.  **调用编码器**: 使用内部的 `MultimodalEncoder` 实例对每个文档的文本内容和/或关联图像进行向量化，生成特征向量。
    3.  **存储元数据**: 将文档的原始信息（如原始ID、文本内容、图像文件路径）存储在 SQLite 数据库中。
        数据库中会为每个文档生成一个自增的整数主键 `internal_id`，这个ID将用作 Faiss 索引中对应向量的唯一标识符。
    4.  **构建和管理向量索引**:
        -   创建并维护 **三个独立** 的 Faiss 索引：一个用于存储纯文本向量，一个用于存储纯图像向量，一个用于存储文本和图像结合的平均向量。
        -   每个 Faiss 索引都使用 `IndexIDMap2` 类型，这允许我们将向量与我们自定义的 `internal_id` (来自SQLite数据库) 关联起来，方便后续检索和数据回溯。
        -   索引使用内积 (`IndexFlatIP`) 作为相似度度量方法。由于所有向量都经过了L2归一化，内积等价于余弦相似度，值越大表示越相似。
    5.  **持久化**: 能够从指定文件路径加载先前已保存的索引文件和数据库，或者在首次运行时创建它们。在关闭时，会将当前的索引状态保存到文件，以便下次使用。

    这种分离索引的设计（文本、图像、平均）允许在检索阶段根据用户查询的类型（纯文本、纯图像、或图文多模态）灵活地选择最合适的索引进行搜索，从而提高检索的准确性和效率。
    我的目标是构建一个可靠的、易于管理的知识库。
    """

    def __init__(
        self,
        db_path: str,
        faiss_text_index_path: str,
        faiss_image_index_path: str,
        faiss_mean_index_path: str,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        初始化 Indexer 实例。我需要一丝不苟地设置好数据库和所有索引文件路径，并准备好编码器。

        Args:
            db_path (str): 指定 SQLite 数据库文件的保存路径。
            faiss_text_index_path (str): 指定文本向量 Faiss 索引文件的保存路径。
            faiss_image_index_path (str): 指定图像向量 Faiss 索引文件的保存路径。
            faiss_mean_index_path (str): 指定平均向量 Faiss 索引文件的保存路径。
            clip_model_name (str): 传递给内部 `MultimodalEncoder` 的 CLIP 模型名称。
                                   此模型名称必须与后续用于查询编码的模型保持一致，以确保向量空间的一致性。

        Raises:
            Exception: 如果在初始化内部编码器、数据库或 Faiss 索引时发生任何错误，则抛出异常。
                       这些都是 Indexer 工作的核心依赖，任何失败都是严重问题。
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info("开始初始化 Indexer...")

        # 保存传入的路径和模型名称配置。清晰的路径信息是管理文件的重要前提。
        self.db_path = db_path
        self.faiss_text_index_path = faiss_text_index_path
        self.faiss_image_index_path = faiss_image_index_path
        self.faiss_mean_index_path = faiss_mean_index_path
        self.logger.info(f"  数据库路径: {self.db_path}")
        self.logger.info(f"  文本索引路径: {self.faiss_text_index_path}")
        self.logger.info(f"  图像索引路径: {self.faiss_image_index_path}")
        self.logger.info(f"  平均向量索引路径: {self.faiss_mean_index_path}")

        # 步骤 1: 初始化多模态编码器 (MultimodalEncoder)。
        # Indexer 内部拥有一个 Encoder 实例，专门用于对其接收的文档进行编码。这是将数据转换为向量的基础。
        self.logger.info(
            f"  - 正在初始化内部 MultimodalEncoder，使用 CLIP 模型: {clip_model_name}..."
        )
        try:
            self.encoder = MultimodalEncoder(clip_model_name)  # 创建编码器实例。
            self.vector_dimension = (
                self.encoder.vector_dimension
            )  # 从编码器获取产生的向量维度。确保所有索引都使用正确的维度。
            self.logger.info(
                f"  - MultimodalEncoder 初始化完成。特征向量维度为: {self.vector_dimension}。"
            )
        except Exception as e_encoder:
            self.logger.critical(
                f"Indexer 初始化严重失败：内部 MultimodalEncoder 创建失败。错误: {e_encoder}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Indexer 无法初始化 Encoder: {e_encoder}"
            ) from e_encoder

        # 步骤 2: 初始化 SQLite 数据库 (用于存储文档元数据)。
        # 调用私有方法 `_init_db` 来确保数据库文件存在，并创建所需的表结构（如果尚不存在）。数据库是元数据的可靠来源。
        self.logger.info(f"  - 正在初始化 SQLite 数据库，路径: '{self.db_path}'...")
        try:
            self._init_db()  # 此方法会处理数据库目录的创建。
            self.logger.info(f"  - SQLite 数据库初始化完成。")
        except Exception as e_db_init:
            self.logger.critical(
                f"Indexer 初始化严重失败：SQLite 数据库初始化失败。错误: {e_db_init}",
                exc_info=True,
            )
            raise RuntimeError(f"Indexer 无法初始化数据库: {e_db_init}") from e_db_init

        # 步骤 3: 加载或创建三个独立的 Faiss 向量索引。
        # 分别为文本向量、图像向量和平均向量（文本+图像组合）加载或创建 Faiss 索引。
        # `_load_or_create_faiss_index` 方法会处理文件存在性检查、维度匹配和新索引创建的逻辑。索引是快速检索的基础，必须准备好。
        self.logger.info(f"  - 正在加载或创建 Faiss 向量索引...")
        try:
            self.text_index = self._load_or_create_faiss_index(
                self.faiss_text_index_path, "文本(Text)"
            )
            self.image_index = self._load_or_create_faiss_index(
                self.faiss_image_index_path, "图像(Image)"
            )
            self.mean_index = self._load_or_create_faiss_index(
                self.faiss_mean_index_path, "平均(Mean)"
            )
            self.logger.info(f"  - 所有 Faiss 索引均已准备就绪。")
        except Exception as e_faiss_init:
            self.logger.critical(
                f"Indexer 初始化严重失败：一个或多个 Faiss 索引加载/创建失败。错误: {e_faiss_init}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Indexer 无法初始化 Faiss 索引: {e_faiss_init}"
            ) from e_faiss_init

        self.logger.info("Indexer 初始化成功完成。我将竭尽全力确保索引的准确和高效。")

    def _init_db(self):
        """
        初始化 SQLite 数据库连接并创建所需的 'documents' 表（如果它还不存在）。
        这个表用于存储文档的元数据，并将原始文档 ID (doc_id) 映射到数据库生成的
        自增主键 `internal_id`。这个 `internal_id` 将作为 Faiss 索引中对应向量的 ID。
        此方法还会确保数据库文件所在的目录存在。这是构建可靠知识库的基石。
        """
        self.logger.info(f"正在连接并初始化数据库表结构于路径: '{self.db_path}'...")

        # 确保数据库文件所在的目录存在，如果不存在则创建它。这是文件系统操作的标准防御性编程。
        db_directory = os.path.dirname(self.db_path)
        if db_directory and not os.path.exists(db_directory):
            try:
                os.makedirs(
                    db_directory, exist_ok=True
                )  # exist_ok=True 表示如果目录已存在则不抛出错误。
                self.logger.debug(
                    f"已确保数据库目录 '{db_directory}' 存在 (或已创建)。"
                )
            except OSError as e:
                self.logger.error(
                    f"创建数据库目录 '{db_directory}' 失败: {e}", exc_info=True
                )
                raise  # Re-raise the exception as this is critical

        try:
            # 使用 'with' 语句确保数据库连接在使用后自动关闭，并能自动处理事务（默认提交，出错回滚）。这是安全的数据库操作模式。
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()  # 获取数据库游标，用于执行 SQL 命令。

                # SQL 语句，用于创建 'documents' 表。
                # `IF NOT EXISTS` 确保如果表已经存在，则不会尝试重新创建它，从而避免错误。
                # 表结构定义必须精确：
                #   - internal_id: 整数类型，主键，自动增长。这是数据库内部ID，也将用作Faiss索引的ID。
                #   - doc_id: 文本类型，唯一约束，不能为空。这是原始文档的唯一标识符 (例如来自JSON的'name'字段)，必须保证唯一性。
                #   - text: 文本类型，存储文档的文本内容，允许为空 (NULL)。
                #   - image_path: 文本类型，存储关联图像文件的路径，允许为空。
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT UNIQUE NOT NULL,
                        text TEXT,
                        image_path TEXT
                    )
                """
                )

                # 可选：在 'doc_id' 列上创建一个索引。
                # 这可以加快通过原始 `doc_id` 查找记录的速度，例如在 `index_documents` 方法中检查重复文档时。索引能提升性能。
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id)"
                )

                conn.commit()  # 提交事务，使表结构更改和索引创建生效。
                self.logger.info(
                    f"数据库表 'documents' (及索引 'idx_doc_id') 初始化成功，或已存在。"
                )
        except sqlite3.Error as e:  # Catch specific SQLite errors
            self.logger.critical(
                f"严重错误：初始化 SQLite 数据库 '{self.db_path}' 失败。错误详情: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"SQLite数据库操作失败: {e}"
            ) from e  # Re-raise as a more generic runtime error
        except Exception as e_general:
            self.logger.critical(
                f"初始化 SQLite 数据库 '{self.db_path}' 时发生未知错误。错误详情: {e_general}",
                exc_info=True,
            )
            raise RuntimeError(
                f"SQLite数据库初始化未知错误: {e_general}"
            ) from e_general

    def _load_or_create_faiss_index(
        self, index_path: str, index_type_description: str
    ) -> faiss.Index:
        """
        尝试从指定路径加载一个 Faiss 索引文件。
        - 如果文件存在且其内部存储的向量维度与当前编码器 (`self.encoder`) 的输出维度匹配，则加载该索引。
        - 如果文件不存在，或者文件存在但维度不匹配（表明该索引可能是用不同模型创建的，不能兼容），则创建一个新的、空的 Faiss 索引。
        - 使用 `faiss.IndexIDMap2` 类型的索引，它允许我们将自定义的 64 位整数 ID (数据库的 internal_id) 与每个向量关联起来。
        此方法还会确保索引文件所在的目录存在。我的职责是确保索引可用，无论是加载旧的还是创建一个新的。

        Args:
            index_path (str): Faiss 索引文件的期望路径。
            index_type_description (str): 索引类型的描述性名称 (例如 "文本", "图像", "平均")，主要用于日志记录。

        Returns:
            faiss.Index: 加载的或新创建的 Faiss 索引对象 (具体类型为 `faiss.IndexIDMap2`)。
        """
        self.logger.info(
            f"正在为 '{index_type_description}' 索引加载或创建 Faiss 文件于路径: '{index_path}'..."
        )

        # 确保 Faiss 索引文件所在的目录存在，如果不存在则创建它。
        index_directory = os.path.dirname(index_path)
        if index_directory and not os.path.exists(index_directory):
            try:
                os.makedirs(index_directory, exist_ok=True)
                self.logger.debug(
                    f"已确保 '{index_type_description}' 索引的目录 '{index_directory}' 存在 (或已创建)。"
                )
            except OSError as e:
                self.logger.critical(
                    f"创建Faiss索引目录 '{index_directory}' 失败: {e}", exc_info=True
                )
                raise  # Re-raise as this is critical

        try:
            # 检查指定的索引文件是否已经存在于文件系统中。
            if os.path.exists(index_path) and os.path.isfile(index_path):  # 确保是文件
                self.logger.info(
                    f"发现已存在的 '{index_type_description}' Faiss 索引文件，尝试加载: {index_path}"
                )
                # 使用 faiss.read_index 函数读取磁盘上的索引文件。
                index = faiss.read_index(index_path)
                self.logger.info(
                    f"文件 '{index_path}' 读取成功，包含 {index.ntotal} 个向量，维度为 {index.d}。"
                )

                # **重要**: 检查加载的索引的维度 (`index.d`) 是否与当前编码器模型产生的向量维度 (`self.vector_dimension`) 一致。
                # 维度不一致意味着索引与当前模型不兼容，强行使用会导致错误或无效结果。必须进行此项检查。
                if index.d != self.vector_dimension:
                    # 如果维度不匹配，这意味着已加载的索引是用不同的（或不同配置的）CLIP 模型创建的，因此不能直接使用。
                    self.logger.warning(
                        f"维度不匹配警告! 加载的 '{index_type_description}' 索引维度 ({index.d}) 与当前编码器配置的维度 ({self.vector_dimension}) 不一致。"
                    )
                    self.logger.warning(
                        f"这通常意味着之前的索引是用不同的模型创建的。将忽略已加载的旧索引，并创建一个新的空 '{index_type_description}' 索引。"
                    )
                    # 创建一个新的、空的 Faiss 索引来替换掉加载的不兼容的旧索引。
                    index = self._create_new_faiss_index(index_type_description)
                else:
                    # 维度匹配，加载成功。太好了，我们可以复用之前的索引了。
                    self.logger.info(
                        f"成功加载 '{index_type_description}' Faiss 索引，维度 ({index.d}) 与当前模型匹配。索引中包含 {index.ntotal} 个向量。"
                    )
            else:
                # 如果索引文件不存在。
                self.logger.info(
                    f"未找到 '{index_type_description}' Faiss 索引文件: '{index_path}'。将创建一个新的空索引。"
                )
                # 调用内部方法创建新的空索引。
                index = self._create_new_faiss_index(index_type_description)
        except Exception as e:
            # 处理在加载或读取索引文件过程中可能发生的任何其他错误。即使加载失败，也要保证能创建一个新索引。
            self.logger.error(
                f"错误：加载或处理 '{index_type_description}' Faiss 索引 '{index_path}' 失败。错误详情: {e}",
                exc_info=True,
            )
            self.logger.info(
                f"作为安全回退机制，将创建一个新的空 '{index_type_description}' 索引。"
            )
            # 即使加载失败，也创建一个新的空索引，以保证程序能够继续运行（尽管可能没有历史数据）。
            index = self._create_new_faiss_index(index_type_description)
        return index

    def _create_new_faiss_index(self, index_type_description: str) -> faiss.Index:
        """
        创建一个新的、空的 Faiss 索引。
        该索引被配置为使用内积 (`IndexFlatIP`) 进行相似度搜索，并使用 `IndexIDMap2` 包装器
        来支持为每个向量存储自定义的 64 位整数 ID。
        `IndexFlatIP` 适用于存储原始（未压缩）向量并进行精确的、暴力的内积搜索。
        对于已经 L2 归一化的向量，内积得分等价于余弦相似度。我选择这个类型是因为它简单且对于归一化向量表现良好。

        Args:
            index_type_description (str): 索引类型的描述 (例如 "文本", "图像")，用于日志记录。

        Returns:
            faiss.Index: 新创建的、空的 `faiss.IndexIDMap2` 索引对象。
        """
        self.logger.info(
            f"开始为 '{index_type_description}' 创建一个新的空 Faiss 索引..."
        )
        # 步骤 1: 创建基础索引 (也称为 quantizer，在更复杂的索引类型中作用更明显)。
        # 这里使用 `faiss.IndexFlatIP`:
        #   - `IndexFlat`: 表示 Faiss 将存储完整的、未经压缩或量化的原始向量。这提供了最精确的搜索结果，但需要更多内存。
        #   - `IP` (Inner Product): 表示该索引将使用内积作为向量间的距离/相似度度量。
        #   当存储的向量都经过 L2 归一化时，它们之间的内积值等于它们之间的余弦相似度。
        #   `self.vector_dimension` 是从 CLIP 模型获取的特征向量的维度，这是索引创建的基础。
        quantizer = faiss.IndexFlatIP(self.vector_dimension)
        self.logger.debug(
            f"  为 '{index_type_description}' 创建了 IndexFlatIP 基础索引，维度: {self.vector_dimension}。"
        )

        # 步骤 2: 创建 ID 映射包装器 `faiss.IndexIDMap2`。
        #   - `IndexIDMap2` 包装了一个基础索引 (此处是 `quantizer`)。
        #   - 它允许我们在向索引添加向量时，为每个向量指定一个我们自己定义的 64 位整数 ID。
        #   - 在搜索时，它会返回这些我们指定的 ID，而不是 Faiss 内部的连续行号。
        #   - '2' 在名称中通常表示它使用了更现代或更灵活的内部ID重映射机制。
        #   - 我们将使用从 SQLite 数据库生成的 `internal_id` 作为这个自定义 ID。这保证了向量与元数据的关联。
        index = faiss.IndexIDMap2(quantizer)
        self.logger.debug(
            f"  将 IndexFlatIP 包装在 IndexIDMap2 中，以支持自定义向量 ID。"
        )

        self.logger.info(
            f"已成功为 '{index_type_description}' 创建一个新的、空的 Faiss 索引 (类型: IndexIDMap2 包裹 IndexFlatIP)。"
        )
        self.logger.info(f"    索引维度: {self.vector_dimension}。")
        self.logger.info(
            f"    相似度度量: 内积 (Inner Product) - 对于归一化向量，这等同于余弦相似度。"
        )
        return index

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        核心的文档索引流程。
        该方法接收一个文档列表，对每个文档进行多模态编码（文本和/或图像），
        然后将文档的元数据存储到 SQLite 数据库中，并将生成的特征向量（文本向量、图像向量、平均向量）
        及其对应的数据库内部ID (`internal_id`) 添加到各自的 Faiss 索引中。
        此方法会处理基于 `doc_id` 的重复文档（即，如果一个具有相同 `doc_id` 的文档已存在于数据库中，则跳过它）。
        为了提高效率，向量会分批收集，然后一次性批量添加到 Faiss 索引中。这是确保效率和可靠性的关键。

        Args:
            documents (List[Dict[str, Any]]): 一个字典列表，其中每个字典代表一个待索引的文档。
                                    每个字典应至少包含 'id' (原始文档ID), 'text' (文本内容),
                                    和 'image_path' (关联图像的路径，可能为None) 这几个键。
                                    这通常是 `load_data_from_json_and_associate_images` 函数的输出格式。
        """
        # 检查输入的文档列表是否为空。如果没有任何文档，就没有必要继续。
        if not documents:
            self.logger.info("未提供任何文档进行索引操作。流程结束。")
            return

        self.logger.info(f"开始执行文档索引流程，准备处理 {len(documents)} 个文档...")

        # 初始化列表，用于批量收集需要添加到 Faiss 索引的向量和它们对应的 ID。
        # 分别为文本、图像和平均向量准备独立的批处理列表。批量操作比逐个添加更高效。
        text_vectors_batch: List[np.ndarray] = []  # 存储文本特征向量 (NumPy 数组)。
        text_ids_batch: List[int] = []  # 存储与文本向量对应的 `internal_id` (整数)。
        image_vectors_batch: List[np.ndarray] = []  # 存储图像特征向量。
        image_ids_batch: List[int] = []  # 存储与图像向量对应的 `internal_id`。
        mean_vectors_batch: List[np.ndarray] = []  # 存储平均（文本+图像）特征向量。
        mean_ids_batch: List[int] = []  # 存储与平均向量对应的 `internal_id`。

        # 初始化计数器，用于跟踪索引过程的各种统计数据。清晰的统计数据帮助了解索引的实际情况。
        processed_count = 0  # 成功处理并至少为其生成了一个向量的文档数量。
        skipped_duplicate_count = 0  # 因 `doc_id` 已存在于数据库中而被跳过的文档数量。
        skipped_invalid_input_count = (
            0  # 因输入数据无效（缺少ID或内容）而被跳过的文档数量。
        )
        encoding_failure_count = 0  # 因编码阶段（文本或图像）出错而未能为其生成向量的文档数量 (即使元数据可能已插入)。
        db_check_error_count = 0  # 因检查重复项时数据库出错而被跳过的文档数量。
        db_insert_error_count = 0  # 因数据库插入操作出错而被跳过的文档数量。

        conn: Optional[sqlite3.Connection] = (
            None  # 初始化数据库连接变量，确保在 try...finally 块中可见，用于最终的连接关闭。
        )
        try:
            # 步骤 1: 建立与 SQLite 数据库的连接。
            self.logger.debug(f"正在连接到数据库: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()  # 获取数据库游标。
            # sqlite3 默认在执行 DML 语句 (如 INSERT) 时会自动开始一个事务。
            # 我们将在所有文档处理完毕后，在循环外部统一提交 (commit) 或回滚 (rollback) 事务，
            # 以确保数据库操作的原子性（相对于整个批次而言）。这是保证数据一致性的重要手段。

            # 步骤 2: 遍历每个待索引的文档。
            self.logger.info(f"开始遍历 {len(documents)} 个文档进行处理和编码...")
            for i, doc_data in enumerate(
                documents
            ):  # 使用 enumerate 获取索引和文档数据，便于日志记录和问题定位。
                doc_id = doc_data.get("id")  # 获取原始文档 ID。
                text_content = doc_data.get("text")  # 获取文本内容。
                image_file_path = doc_data.get("image_path")  # 获取图像路径。

                self.logger.debug(f"处理文档 {i+1}/{len(documents)}: ID='{doc_id}'")

                # 基本有效性验证：`doc_id` 必须存在且非空。这是文档的唯一标识。
                doc_id_str = str(doc_id).strip() if doc_id is not None else None
                if not doc_id_str:
                    self.logger.warning(
                        f"跳过列表中的第 {i+1} 条记录（原始索引 {i}），因其缺少有效 'id' 字段。记录: {doc_data}"
                    )
                    skipped_invalid_input_count += 1  # 计入无效输入
                    continue  # 跳到下一个文档。

                # 至少需要文本或图像路径之一才能进行有意义的编码。
                has_valid_text = text_content is not None and str(text_content).strip()
                has_valid_image = (
                    image_file_path is not None and str(image_file_path).strip()
                )
                if not has_valid_text and not has_valid_image:
                    self.logger.warning(
                        f"跳过文档 ID '{doc_id_str}'，因为它既没有有效的文本内容，也没有有效的关联图像路径。无法为其生成任何向量。"
                    )
                    skipped_invalid_input_count += 1  # 计入无效输入
                    continue  # 跳到下一个文档。

                # --- 2a. 检查文档是否已在数据库中存在 (基于 `doc_id`) ---
                # 这是防止重复索引的关键步骤。
                try:
                    cursor.execute(
                        "SELECT internal_id FROM documents WHERE doc_id = ?",
                        (doc_id_str,),
                    )
                    existing_record = (
                        cursor.fetchone()
                    )  # 获取查询结果（如果存在的话）。
                    if existing_record:
                        self.logger.debug(
                            f"文档 ID '{doc_id_str}' 已存在于数据库中 (其 internal_id 为: {existing_record[0]})。将跳过此重复文档的索引。"
                        )
                        skipped_duplicate_count += 1
                        continue  # 跳到下一个文档。
                except sqlite3.Error as e_check:
                    self.logger.error(
                        f"检查文档 ID '{doc_id_str}' 是否存在时发生数据库错误: {e_check}。将跳过此文档以防意外行为。"
                    )
                    db_check_error_count += 1  # 计入数据库检查错误
                    continue  # 跳到下一个文档以确保安全。

                # --- 2b. 使用内部 Encoder 对文档进行多模态向量化 ---
                # 在插入数据库之前进行编码，如果编码失败，则不插入元数据，保持一致性。
                encoded_data: Optional[Dict[str, Optional[np.ndarray]]] = (
                    None  # 初始化编码结果。
                )
                encoding_succeeded = False
                try:
                    self.logger.debug(f"开始为文档 '{doc_id_str}' 进行多模态编码...")
                    # Pass original text_content and image_file_path, encode() handles validation internally
                    encoded_data = self.encoder.encode(
                        text=text_content, image_path=image_file_path
                    )
                    # 检查编码是否至少生成了一个向量
                    if encoded_data and (
                        encoded_data.get("text_vector") is not None
                        or encoded_data.get("image_vector") is not None
                        or encoded_data.get("mean_vector") is not None
                    ):
                        encoding_succeeded = True
                        self.logger.debug(
                            f"文档 '{doc_id_str}' 编码成功，至少生成了一个向量。"
                        )
                    else:
                        self.logger.warning(
                            f"文档 '{doc_id_str}' 编码完成，但未能生成任何有效向量 (即使输入有效)。"
                        )
                        encoding_failure_count += 1
                        # 不继续插入数据库，因为没有向量可以关联

                except Exception as encode_e:
                    self.logger.error(
                        f"严重错误：在编码文档 '{doc_id_str}' 时发生意外错误: {encode_e}",
                        exc_info=True,
                    )
                    encoding_failure_count += 1
                    # 不继续插入数据库，因为编码失败

                # 如果编码失败（没有生成任何向量或发生异常），则跳过此文档的后续处理（数据库插入和Faiss添加）。
                if not encoding_succeeded:
                    self.logger.warning(
                        f"由于文档 '{doc_id_str}' 的编码未能生成有效向量，将跳过此文档的数据库插入和 Faiss 索引添加。"
                    )
                    continue

                # --- 2c. 将文档元数据插入到数据库，并获取生成的 `internal_id` ---
                # 只有在编码成功后才执行此步骤。
                internal_id: Optional[int] = None  # 初始化 internal_id。
                try:
                    # 执行 INSERT 语句将新文档的元数据插入到 'documents' 表。
                    # 使用参数化查询 (问号占位符) 来防止 SQL 注入攻击。
                    cursor.execute(
                        "INSERT INTO documents (doc_id, text, image_path) VALUES (?, ?, ?)",
                        (
                            doc_id_str,
                            text_content,
                            image_file_path,
                        ),  # 使用清理过的 doc_id_str
                    )
                    # 获取刚刚插入行的自增主键 (`internal_id`)。
                    # `cursor.lastrowid` 返回最后插入行的 ROWID。
                    internal_id = cursor.lastrowid
                    if internal_id is None:
                        # 这是一个理论上不太可能发生但在极端情况下需要考虑的问题。
                        self.logger.error(
                            f"严重数据库错误：为文档 '{doc_id_str}' 插入元数据后，未能获取有效的 internal_id (lastrowid is None)。这将导致向量无法与元数据关联！"
                        )
                        # 尝试回滚当前文档的操作？或者记录错误并继续？这里选择记录并尝试继续，但标记错误。
                        db_insert_error_count += 1
                        continue  # 跳过这个文档的 Faiss 添加，因为没有有效的 ID
                    self.logger.debug(
                        f"文档 '{doc_id_str}' 的元数据已成功插入数据库，获得的 internal_id: {internal_id}"
                    )

                except sqlite3.IntegrityError:
                    # 当尝试插入的 `doc_id` 违反了表的 UNIQUE 约束时（理论上不应发生，因为前面已检查过）。
                    # 这是一个额外的防御层。
                    self.logger.error(
                        f"数据库完整性错误：尝试插入已存在的文档 ID '{doc_id_str}'（可能是并发问题或检查逻辑遗漏）。将跳过此文档。"
                    )
                    # 之前检查过，理论上不应该到这里，但作为安全措施计数。
                    skipped_duplicate_count += 1
                    continue
                except sqlite3.Error as db_e:
                    self.logger.error(
                        f"数据库错误：在为文档 '{doc_id_str}' 插入元数据时发生错误: {db_e}。将跳过此文档的 Faiss 添加。"
                    )
                    db_insert_error_count += 1
                    continue  # 跳过 Faiss 添加，因为元数据插入失败

                # --- 2d. 将成功编码的向量添加到对应的批处理列表中 ---
                # 只有在编码成功 且 数据库插入成功 (获取到internal_id) 后才执行。
                if internal_id is not None and encoded_data is not None:
                    at_least_one_vector_added_for_doc = (
                        False  # 跟踪该文档是否至少有一个向量被添加到批处理。
                    )
                    if encoded_data.get("text_vector") is not None:
                        text_vectors_batch.append(encoded_data["text_vector"])  # type: ignore
                        text_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(
                            f"  文本向量已为文档 '{doc_id_str}' (internal_id: {internal_id}) 准备好加入批处理。"
                        )

                    if encoded_data.get("image_vector") is not None:
                        image_vectors_batch.append(encoded_data["image_vector"])  # type: ignore
                        image_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(
                            f"  图像向量已为文档 '{doc_id_str}' (internal_id: {internal_id}) 准备好加入批处理。"
                        )

                    if encoded_data.get("mean_vector") is not None:
                        mean_vectors_batch.append(encoded_data["mean_vector"])  # type: ignore
                        mean_ids_batch.append(internal_id)
                        at_least_one_vector_added_for_doc = True
                        self.logger.debug(
                            f"  平均向量已为文档 '{doc_id_str}' (internal_id: {internal_id}) 准备好加入批处理。"
                        )

                    # 如果编码成功，数据库插入成功，但没有向量被添加到批处理（理论上不可能，因为encoding_succeeded保证了至少一个向量存在）
                    # 这是一个内部逻辑检查。
                    if not at_least_one_vector_added_for_doc:
                        self.logger.error(
                            f"内部逻辑错误：文档 '{doc_id_str}' (internal_id: {internal_id}) 编码和数据库插入均成功，但没有向量被添加到批处理！"
                        )
                        # 这种情况不应该发生，但如果发生了，也算作处理失败的一种形式
                        encoding_failure_count += 1  # 归类为编码相关问题
                    else:
                        processed_count += 1  # 只有编码成功、DB插入成功、向量准备好加入批处理，才算成功处理。

            # --- 文档遍历和初步处理完成 ---
            self.logger.info(f"所有 {len(documents)} 个输入文档已遍历处理完毕。")
            self.logger.info(f"准备将收集到的向量批量添加到 Faiss 索引中...")
            self.logger.info(f"  - 待添加文本向量数量: {len(text_ids_batch)}")
            self.logger.info(f"  - 待添加图像向量数量: {len(image_ids_batch)}")
            self.logger.info(f"  - 待添加平均向量数量: {len(mean_ids_batch)}")

            # --- 步骤 3: 批量将向量和 ID 添加到对应的 Faiss 索引 ---
            # 批量添加比逐个添加效率高得多。
            faiss_add_errors = 0
            try:
                if text_vectors_batch:
                    ids_np_text = np.array(
                        text_ids_batch, dtype="int64"
                    )  # Faiss IndexIDMap2 需要 int64 类型的 ID。
                    vectors_np_text = np.array(
                        text_vectors_batch, dtype="float32"
                    )  # Faiss 通常使用 float32。
                    self.text_index.add_with_ids(
                        vectors_np_text, ids_np_text
                    )  # 将向量和对应的 ID 批量添加到文本索引。
                    self.logger.info(
                        f"已成功向文本(Text) Faiss 索引批量添加 {len(text_vectors_batch)} 个向量。当前索引总数: {self.text_index.ntotal}"
                    )
            except Exception as faiss_e_text:
                self.logger.error(
                    f"错误：向文本(Text) Faiss 索引批量添加向量时失败: {faiss_e_text}",
                    exc_info=True,
                )
                faiss_add_errors += 1

            try:
                if image_vectors_batch:
                    ids_np_image = np.array(image_ids_batch, dtype="int64")
                    vectors_np_image = np.array(image_vectors_batch, dtype="float32")
                    self.image_index.add_with_ids(vectors_np_image, ids_np_image)
                    self.logger.info(
                        f"已成功向图像(Image) Faiss 索引批量添加 {len(image_vectors_batch)} 个向量。当前索引总数: {self.image_index.ntotal}"
                    )
            except Exception as faiss_e_image:
                self.logger.error(
                    f"错误：向图像(Image) Faiss 索引批量添加向量时失败: {faiss_e_image}",
                    exc_info=True,
                )
                faiss_add_errors += 1

            try:
                if mean_vectors_batch:
                    ids_np_mean = np.array(mean_ids_batch, dtype="int64")
                    vectors_np_mean = np.array(mean_vectors_batch, dtype="float32")
                    self.mean_index.add_with_ids(vectors_np_mean, ids_np_mean)
                    self.logger.info(
                        f"已成功向平均(Mean) Faiss 索引批量添加 {len(mean_vectors_batch)} 个向量。当前索引总数: {self.mean_index.ntotal}"
                    )
            except Exception as faiss_e_mean:
                self.logger.error(
                    f"错误：向平均(Mean) Faiss 索引批量添加向量时失败: {faiss_e_mean}",
                    exc_info=True,
                )
                faiss_add_errors += 1

            # --- 步骤 4: 提交数据库事务 ---
            # 如果 Faiss 添加过程中出现任何错误，也应该提交数据库更改，因为元数据插入是在此之前完成的。
            # 但需要记录 Faiss 添加失败的情况。
            if conn:
                conn.commit()
                self.logger.info("数据库事务已成功提交。元数据更改已持久化。")
                if faiss_add_errors > 0:
                    self.logger.warning(
                        f"警告：虽然数据库事务已提交，但在向 {faiss_add_errors} 个 Faiss 索引添加向量时发生了错误。数据库和 Faiss 索引可能存在不一致！"
                    )

        except Exception as e:
            self.logger.critical(
                f"严重错误：在文档索引过程中发生意外的顶级异常: {e}", exc_info=True
            )
            # 如果在处理过程中发生任何未捕获的异常，回滚数据库事务是必要的，以避免数据库处于不一致状态。
            if conn:
                self.logger.warning(
                    "检测到严重错误，正在尝试回滚数据库事务以撤销本批次未提交的更改..."
                )
                try:
                    conn.rollback()
                    self.logger.info("数据库事务已成功回滚。")
                except Exception as rb_e:
                    self.logger.error(
                        f"错误：尝试回滚数据库事务时失败: {rb_e}", exc_info=True
                    )
        finally:
            # 无论发生什么，最后都要关闭数据库连接。
            if conn:
                conn.close()
                self.logger.debug("数据库连接已关闭。")

        # --- 打印索引过程的最终总结信息 ---
        # 这是一个重要的报告，总结了本次索引操作的结果。
        self.logger.info(f"\n--- 文档索引过程总结 ---")
        self.logger.info(f"- 输入文档总数: {len(documents)}")
        self.logger.info(
            f"- 因输入无效(缺少ID或内容)跳过的文档数: {skipped_invalid_input_count}"
        )
        self.logger.info(
            f"- 因 'doc_id' 在数据库中已存在而跳过的文档数: {skipped_duplicate_count}"
        )
        self.logger.info(
            f"- 因检查重复项时数据库错误跳过的文档数: {db_check_error_count}"
        )
        self.logger.info(
            f"- 因编码未能生成有效向量而跳过的文档数: {encoding_failure_count}"
        )
        self.logger.info(f"- 因数据库插入错误而跳过的文档数: {db_insert_error_count}")
        self.logger.info(
            f"- 成功处理(编码成功+DB插入成功+准备添加到Faiss)的文档数: {processed_count}"
        )
        self.logger.info(
            f"- 向 Faiss 添加向量时发生错误的索引数量: {faiss_add_errors if 'faiss_add_errors' in locals() else 'N/A'}"
        )  # Check if var exists

        # 获取当前索引中的向量总数，使用 getattr 安全访问 ntotal 属性。
        text_final_count = getattr(self.text_index, "ntotal", "N/A")
        image_final_count = getattr(self.image_index, "ntotal", "N/A")
        mean_final_count = getattr(self.mean_index, "ntotal", "N/A")
        self.logger.info(f"- 当前文本 Faiss 索引中的向量总数: {text_final_count}")
        self.logger.info(f"- 当前图像 Faiss 索引中的向量总数: {image_final_count}")
        self.logger.info(f"- 当前平均 Faiss 索引中的向量总数: {mean_final_count}")

        # 获取数据库中的文档总数，并与Faiss索引中的向量数进行比较，检查一致性。
        db_final_count = self.get_document_count()
        self.logger.info(
            f"- 当前 SQLite 数据库中存储的文档元数据记录总数: {db_final_count}"
        )

        # 进行一致性检查
        if (
            isinstance(text_final_count, int)
            and isinstance(image_final_count, int)
            and isinstance(mean_final_count, int)
        ):
            max_faiss_vectors = max(
                text_final_count, image_final_count, mean_final_count
            )
            # 理想情况下，成功处理的文档数 (processed_count) 应该等于添加到 Faiss 的向量数（对于存在对应向量类型的文档）
            # 数据库的最终记录数 (db_final_count) 应该等于处理过程中成功插入数据库的总数。
            # 注意：processed_count 是本轮成功处理的数量，而 db_final_count 和 Faiss count 是累积总量。
            # 更准确的检查是比较本轮成功添加到 Faiss 的批次大小与 processed_count。

            # 粗略检查：比较数据库总数和Faiss最大总数
            if (
                db_final_count > max_faiss_vectors and max_faiss_vectors >= 0
            ):  # Allow max_faiss_vectors == 0
                self.logger.warning(
                    f"数据一致性提示：数据库记录数 ({db_final_count}) 多于 Faiss 索引中的最大向量数 ({max_faiss_vectors})。"
                )
                self.logger.warning(
                    f"                 这可能是正常的（例如，部分文档只有文本没有图像，或反之），但也可能表示部分向量未能添加成功。请检查日志。"
                )
            elif db_final_count < max_faiss_vectors:
                self.logger.error(
                    f"数据一致性错误：数据库记录数 ({db_final_count}) 少于某个 Faiss 索引中的最大向量数 ({max_faiss_vectors})。数据可能存在严重不一致！"
                )
                self.logger.error(
                    f"                 这可能意味着部分文档的向量被添加到了Faiss，但其元数据未能插入数据库！请检查错误日志。"
                )
        else:
            self.logger.warning(
                "未能执行完整的数据库/Faiss数量一致性检查，因为无法获取所有索引的数值计数。"
            )

        self.logger.info(f"--- 文档索引过程结束 ---")

    def get_document_by_internal_id(self, internal_id: int) -> Optional[Dict[str, Any]]:
        """
        根据 Faiss 搜索返回的 `internal_id` (即数据库中的主键)，从 SQLite 数据库中检索对应的原始文档元数据。
        这是从向量检索结果回溯到原始文档信息的必要步骤。

        Args:
            internal_id (int): 要查询的文档在数据库中的 `internal_id` (通常由 Faiss 搜索返回)。

        Returns:
            Optional[Dict[str, Any]]: 如果找到文档，则返回一个包含文档信息的字典。
                            该字典通常包含 'id' (原始 doc_id), 'text', 'image_path', 和 'internal_id'。
                            如果数据库中找不到具有该 `internal_id` 的记录，则返回 None。
        """
        self.logger.debug(
            f"尝试从数据库根据 internal_id '{internal_id}' 获取文档元数据..."
        )
        try:
            # 连接到 SQLite 数据库。
            with sqlite3.connect(self.db_path) as conn:
                # 设置 conn.row_factory = sqlite3.Row 使得查询结果可以像字典一样通过列名访问，更方便。
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # 执行 SELECT 查询，根据 internal_id 查找记录。
                cursor.execute(
                    "SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id = ?",
                    (internal_id,),  # 注意参数是一个元组。
                )
                row = cursor.fetchone()  # 获取一行结果。

                if row:
                    # 如果找到了记录 (row 不为 None)。
                    doc_data = dict(
                        row
                    )  # 将 sqlite3.Row 对象转换为标准的 Python 字典。
                    # 为了与外部接口或数据结构保持一致（例如，原始输入时的 'id'），
                    # 将从数据库中取出的 'doc_id' 键重命名为 'id'。
                    doc_data["id"] = doc_data.pop("doc_id")
                    self.logger.debug(
                        f"成功为 internal_id '{internal_id}' 找到文档元数据: ID='{doc_data['id']}'"
                    )
                    return doc_data  # 返回包含文档信息的字典。
                else:
                    # 如果没有找到具有该 internal_id 的记录。
                    self.logger.warning(
                        f"未能在数据库中找到 internal_id 为 '{internal_id}' 的文档元数据。"
                    )
                    return None  # 返回 None。
        except sqlite3.Error as e_sql:
            self.logger.error(
                f"数据库错误：从数据库根据 internal_id '{internal_id}' 获取文档时发生错误: {e_sql}",
                exc_info=True,
            )
            return None
        except Exception as e_general:
            self.logger.error(
                f"未知错误：从数据库根据 internal_id '{internal_id}' 获取文档时发生: {e_general}",
                exc_info=True,
            )
            return None

    def get_documents_by_internal_ids(
        self, internal_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        根据一个 `internal_id` 的列表，从 SQLite 数据库中批量检索对应的多个文档的元数据。
        使用批量查询 (SELECT ... WHERE internal_id IN (...)) 通常比多次单独查询更高效，
        尤其是在处理 Faiss 返回的 Top-K 结果列表时。这是为了提高检索后获取元数据的效率。

        Args:
            internal_ids (List[int]): 一个包含多个数据库 `internal_id` 的整数列表。

        Returns:
            Dict[int, Dict[str, Any]]: 一个字典，其中键是 `internal_id`，值是对应的文档数据字典
                             (通常包含 'id', 'text', 'image_path', 'internal_id')。
                             如果列表中的某个 ID 在数据库中找不到，则结果字典中不会包含该 ID 的条目。
                             如果输入的 `internal_ids` 列表为空，则返回一个空字典。
        """
        # 如果输入的 ID 列表为空，直接返回空字典，无需查询数据库。
        if not internal_ids:
            self.logger.debug(
                "请求批量获取文档，但提供的 internal_id 列表为空。返回空结果。"
            )
            return {}

        # 限制一次批量查询的ID数量，避免SQL语句过长导致的问题。
        # SQLite 默认的 SQLITE_MAX_VARIABLE_NUMBER 是 999，但可以被编译时修改。
        # 为保险起见，设置一个稍小的值，或者如果需要处理大量 ID，应实现分块查询逻辑。
        max_ids_per_query = 900  # Use a slightly conservative limit
        results: Dict[int, Dict[str, Any]] = {}  # 初始化结果字典。

        # 对 ID 进行分块处理，以防列表过长
        for i in range(0, len(internal_ids), max_ids_per_query):
            id_chunk = internal_ids[i : i + max_ids_per_query]
            if not id_chunk:
                continue  # Skip empty chunks (shouldn't happen with correct slicing)

            self.logger.debug(
                f"尝试从数据库根据 internal_id 列表块 (块大小 {len(id_chunk)}) 批量获取文档元数据..."
            )

            try:
                # 连接到 SQLite 数据库。
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row  # 设置行工厂，方便处理查询结果。
                    cursor = conn.cursor()

                    # 构建 SQL 查询语句，使用 IN 操作符和参数占位符进行批量查询。
                    # 1. 创建占位符字符串: "(?, ?, ..., ?)" - 每个 ID 对应一个 '?'。
                    placeholders = ",".join("?" for _ in id_chunk)
                    # 2. 构建完整的 SQL 查询语句。
                    query = f"SELECT internal_id, doc_id, text, image_path FROM documents WHERE internal_id IN ({placeholders})"
                    self.logger.debug(
                        f"执行批量查询SQL (块 {i // max_ids_per_query + 1}): {query[:100]}... (参数数量: {len(id_chunk)})"
                    )

                    # 执行查询，将 ID 列表块作为参数传递给 execute 方法。
                    cursor.execute(query, id_chunk)
                    rows = cursor.fetchall()  # 获取所有匹配的行。
                    self.logger.debug(f"数据库批量查询块返回了 {len(rows)} 行记录。")

                    # 遍历查询结果。
                    for row in rows:
                        doc_data = dict(row)  # 将 sqlite3.Row 转换为字典。
                        doc_data["id"] = doc_data.pop(
                            "doc_id"
                        )  # 重命名 'doc_id' 为 'id'。
                        # 使用 internal_id 作为键，将文档数据存入结果字典。
                        results[doc_data["internal_id"]] = doc_data

                    # 检查当前块中是否有ID未找到 (如果 len(rows) < len(id_chunk))，并记录警告。
                    if len(rows) < len(id_chunk):
                        found_ids_in_chunk_set = set(
                            row["internal_id"] for row in rows
                        )  # Use set for efficiency
                        missing_ids_in_chunk = [
                            id_val
                            for id_val in id_chunk
                            if id_val not in found_ids_in_chunk_set
                        ]
                        if missing_ids_in_chunk:
                            self.logger.warning(
                                f"在批量获取文档块时，以下 internal_id 未在数据库中找到: {missing_ids_in_chunk}"
                            )
                            # Removed the second warning line to reduce noise, the implication is clear.

            except sqlite3.Error as e_sql:
                self.logger.error(
                    f"数据库错误：从数据库根据 internal_id 列表块获取文档时发生: {e_sql}",
                    exc_info=True,
                )
                # 返回当前已收集的结果，并可能跳过后续块（或可以设计为继续尝试其他块）。
                # 这里选择继续处理下一个块，但错误已被记录。
            except Exception as e_general:
                self.logger.error(
                    f"未知错误：从数据库根据 internal_id 列表块获取文档时发生: {e_general}",
                    exc_info=True,
                )
                # 同上，记录错误并尝试继续。

        self.logger.debug(
            f"批量获取文档元数据完成，共处理 {len(internal_ids)} 个请求 ID，返回 {len(results)} 个文档的信息。"
        )
        # 最后再检查一次总数是否匹配，以防分块逻辑隐藏问题。
        if len(results) < len(internal_ids):
            all_requested_ids_set = set(internal_ids)
            all_found_ids_set = set(results.keys())
            final_missing_ids = list(all_requested_ids_set - all_found_ids_set)
            if final_missing_ids:
                self.logger.warning(
                    f"最终检查：在所有请求的 internal_id 中，以下 ID 未在数据库中找到: {final_missing_ids[:20]}{'...' if len(final_missing_ids)>20 else ''} (总计缺失 {len(final_missing_ids)} 个)"
                )
                self.logger.warning(
                    "  这可能表示 Faiss 索引与数据库元数据之间存在不一致。请检查日志和数据源。"
                )

        return results

    def get_document_count(self) -> int:
        """
        获取当前 SQLite 数据库 'documents' 表中存储的文档总数量。
        这是一个简单的计数功能，但对于监控知识库大小非常有用。

        Returns:
            int: 数据库中 'documents' 表的总行数。如果发生错误，则返回 0。
        """
        self.logger.debug(f"开始从数据库 '{self.db_path}' 获取文档总数...")
        try:
            # 连接到 SQLite 数据库。
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 执行 COUNT(*) 查询获取总行数。
                cursor.execute("SELECT COUNT(*) FROM documents")
                # fetchone() 返回一个包含单个值的元组，例如 (50,)。
                count_result = cursor.fetchone()
                # 提取元组中的计数值。如果查询无结果（理论上 COUNT(*) 总有结果，但做个健壮性检查），则默认为 0。
                count = (
                    count_result[0]
                    if count_result and count_result[0] is not None
                    else 0
                )
                self.logger.debug(f"数据库中文档总数为: {count}")
                return count
        except sqlite3.Error as e_sql:
            self.logger.error(
                f"数据库错误：从数据库获取文档总数时发生错误: {e_sql}", exc_info=True
            )
            return 0
        except Exception as e_general:
            self.logger.error(
                f"未知错误：从数据库获取文档总数时发生: {e_general}", exc_info=True
            )
            return 0

    def save_indices(self):
        """
        将内存中的所有三个 Faiss 索引（文本、图像、平均）分别保存到它们对应的文件路径中。
        这个方法用于持久化索引的状态，以便在下次程序启动时可以加载这些索引，从而避免重新处理和编码所有文档。
        只有当索引非空（即包含至少一个向量）时，才会执行保存操作。确保辛苦构建的索引不会丢失。
        """
        self.logger.info("开始尝试将所有 Faiss 索引保存到磁盘文件...")
        # 调用内部辅助方法 `_save_single_index` 分别保存每个索引。
        # 传递索引对象、目标文件路径和索引类型描述（用于日志）。
        if hasattr(self, "text_index"):  # 检查对象是否存在且有效
            self._save_single_index(
                self.text_index, self.faiss_text_index_path, "文本(Text)"
            )
        else:
            self.logger.warning("文本(Text)索引对象不存在，无法保存。")

        if hasattr(self, "image_index"):
            self._save_single_index(
                self.image_index, self.faiss_image_index_path, "图像(Image)"
            )
        else:
            self.logger.warning("图像(Image)索引对象不存在，无法保存。")

        if hasattr(self, "mean_index"):
            self._save_single_index(
                self.mean_index, self.faiss_mean_index_path, "平均(Mean)"
            )
        else:
            self.logger.warning("平均(Mean)索引对象不存在，无法保存。")

        self.logger.info(
            "所有 Faiss 索引的保存操作已完成（或已跳过空索引/不存在的索引）。"
        )

    def _save_single_index(
        self, index: Optional[faiss.Index], index_path: str, index_type_description: str
    ):
        """
        辅助方法：保存单个 Faiss 索引到指定的文件路径。
        仅当索引对象有效且包含至少一个向量时才执行保存。
        此方法还会确保索引文件要保存到的目录存在。每一个细节都不能忽略。

        Args:
            index (Optional[faiss.Index]): 需要保存的 Faiss 索引对象。可能是 None。
            index_path (str): 保存索引的目标文件完整路径。
            index_type_description (str): 索引类型的描述性名称 (例如 "文本", "图像")，用于日志记录。
        """
        self.logger.debug(
            f"准备保存 '{index_type_description}' Faiss 索引到路径: '{index_path}'..."
        )

        if index is None:
            self.logger.warning(
                f"  警告：'{index_type_description}' Faiss 索引对象为 None，无法执行保存操作。"
            )
            return

        # 检查索引对象是否有效（存在 `ntotal` 属性，表示向量数量）以及向量数量是否大于 0。只有非空索引才有保存的价值。
        if hasattr(index, "ntotal") and index.ntotal > 0:
            try:
                # 确保索引文件要保存到的目录存在，如果不存在则创建它。
                index_directory = os.path.dirname(index_path)
                if index_directory and not os.path.exists(index_directory):
                    os.makedirs(index_directory, exist_ok=True)
                    self.logger.debug(
                        f"  已确保 '{index_type_description}' 索引的保存目录 '{index_directory}' 存在 (或已创建)。"
                    )

                # 使用 faiss.write_index 函数将内存中的索引对象写入到指定的磁盘文件。
                faiss.write_index(index, index_path)
                self.logger.info(
                    f"  成功：'{index_type_description}' Faiss 索引 (包含 {index.ntotal} 个向量) 已保存到: {index_path}"
                )
            except Exception as e:
                # 处理在保存索引过程中可能发生的错误 (例如，磁盘空间不足、文件写入权限问题)。
                self.logger.error(
                    f"  错误：保存 '{index_type_description}' Faiss 索引到 '{index_path}' 失败。错误详情: {e}",
                    exc_info=True,
                )
        elif hasattr(index, "ntotal"):  # 索引存在但为空 (ntotal == 0)
            self.logger.info(
                f"  跳过：'{index_type_description}' Faiss 索引为空 (ntotal={index.ntotal})，因此不保存到 '{index_path}'。"
            )
        else:  # 索引对象无效或未正确初始化
            self.logger.warning(
                f"  警告：'{index_type_description}' Faiss 索引似乎未正确初始化 (缺少 ntotal 属性)，无法执行保存操作。"
            )

    def close(self):
        """
        关闭 Indexer 实例时调用的清理方法。
        主要职责是确保所有内存中的 Faiss 索引都已尝试保存到磁盘。
        SQLite 数据库连接是通过 `with sqlite3.connect(...)` 语句管理的，在每个相关方法结束时会自动关闭，
        因此这里不需要显式关闭数据库连接。
        Faiss 索引对象本身在 Python 中是内存对象，它们不需要像文件句柄那样显式关闭；保存它们的状态即是“关闭”操作。
        这是一个负责任的结束流程，确保资源得到妥善处理。
        """
        self.logger.info("开始关闭 Indexer 实例...")
        # 调用 save_indices 方法，确保存储所有 Faiss 索引的最新状态。
        self.save_indices()
        self.logger.info("Indexer 实例关闭完成。所有 Faiss 索引已尝试保存。")
