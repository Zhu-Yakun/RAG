from typing import (
    List,
    Dict,
    Optional,
    Any,
)  # 导入类型提示模块。使用类型提示能让代码更清晰、更易于理解和维护，也能帮助静态分析工具发现潜在错误，这是高质量代码的重要保障。(增加了 Any 类型，以适应某些字典中可能包含的更广泛的数据类型)
import logging  # 导入日志模块。这是追踪程序运行状态、诊断问题、记录信息、警告和错误的核心工具。详细的日志是确保系统可维护性的基石。

# -------------------------------------------------------------------------------------------------
# 全局日志记录器设置 (在 `if __name__ == "__main__":` 中会进一步精细配置，这里只是初始化)
# 这是一个重要的工具，我必须确保它随时可用，以便记录系统的每一个动作和潜在问题。
# -------------------------------------------------------------------------------------------------
logger = logging.getLogger(
    __name__
)  # 初始化一个模块级别的日志记录器实例。`__name__` 会被设置成当前模块的名称，便于区分日志来源。
import os  # 导入操作系统模块。它提供了与操作系统交互的必要功能，例如处理文件路径、检查文件或目录是否存在、以及创建目录等。这些操作对于管理索引和数据文件至关重要。

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import zhipuai  # 导入 ZhipuAI 客户端库。用于与智谱 AI 开发的大语言模型 (LLM) API 进行交互。它将负责根据检索到的信息生成最终答案。

# 安装提示: pip install zhipuai。


# -------------------------------------------------------------------------------------------------
# 生成器类 (Generator)
# Generator 是系统的“问答大脑”，它根据检索到的信息生成用户最终看到的自然语言答案。
# 它的表现依赖于底层的LLM能力和构建Prompt的质量。
# -------------------------------------------------------------------------------------------------
class Generator:
    """
    Generator 类负责与大语言模型 (LLM) API (此处特指 ZhipuAI 的 API) 进行交互，
    以根据用户查询和检索到的上下文信息生成最终的自然语言答案。

    其核心工作流程包括：
    1.  **构建提示 (Prompt)**: 将用户的原始查询和由 Retriever 检索到的相关文档上下文列表，
        组合成一个结构化的提示 (prompt)。这个提示会指导 LLM 如何根据提供的上下文来回答问题，
        并遵循特定的规则（例如，要求 LLM 仅基于提供的上下文作答、如何处理信息不足的情况等）。
    2.  **调用 LLM API**: 将构建好的提示发送给指定的 ZhipuAI 大语言模型 API。
    3.  **处理响应**: 对 LLM API 返回的原始文本响应进行基本的后处理（例如，去除多余的空白字符）。

    此类依赖于 `zhipuai` Python 库来与 ZhipuAI API 进行通信，并且需要一个有效的 API Key。
    API Key 的获取优先级如下：
    - 首先，从构造函数参数 `api_key` 获取。
    - 如果构造函数未提供，则尝试从环境变量 `ZHIPUAI_API_KEY` 读取。
    如果没有有效的 API Key，Generator 将无法工作。我必须确保 API Key 的可用性。
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "glm-4-flash"):
        """
        初始化 Generator 实例。我需要确保能够成功连接到智谱AI的API服务。

        Args:
            api_key (Optional[str]): ZhipuAI 的 API Key。如果在此处提供，将优先使用这个 Key。
                                     如果为 None，则会尝试从环境变量 `ZHIPUAI_API_KEY` 中读取。
                                     API Key 是调用服务的凭证，必须确保获取到。
            model_name (str): 指定要调用的 ZhipuAI 平台的模型名称。例如 "glm-4-flash", "glm-4" 等。
                              不同的模型具有不同的能力、速度、上下文窗口大小和调用成本。
                              默认值为 "glm-4-flash"，这是一个速度较快且性价比较高的模型，适合示例使用。
                              请查阅 ZhipuAI 官方文档以获取最新的可用模型列表和特性。

        Raises:
            ValueError: 如果 `api_key` 参数为 None 并且在环境变量 `ZHIPUAI_API_KEY` 中也找不到有效的 Key。
                        没有 API Key，Generator 将无法与 ZhipuAI 服务通信，这是致命的初始化失败。
            RuntimeError: 如果 ZhipuAI 客户端在初始化过程中发生其他错误 (例如，网络问题、`zhipuai`库安装问题等)。
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.info(f"开始初始化 Generator，准备使用 ZhipuAI 模型: {model_name}")

        # 决定最终使用的 API Key：优先使用通过参数传入的，否则尝试从环境变量获取。
        final_api_key = api_key if api_key else os.getenv("ZHIPUAI_API_KEY")

        # 检查是否成功获取到 API Key。如果获取不到，必须报错并终止初始化。
        if not final_api_key:
            error_message = (
                "Generator 初始化错误: ZhipuAI API Key 未提供。\n"
                "调用大语言模型需要有效的 API Key。请通过以下方式之一提供 API Key：\n"
                "  1. 在初始化 Generator 时，通过 'api_key' 参数传入。\n"
                "  2. 将 API Key 设置到名为 'ZHIPUAI_API_KEY' 的环境变量中。"
            )
            self.logger.critical(error_message)
            raise ValueError(error_message)
        else:
            # 为了安全，不记录完整的 API Key，只记录获取成功。
            self.logger.info("成功获取到 ZhipuAI API Key (来源可能是参数或环境变量)。")

        try:
            # 初始化 ZhipuAI 客户端。
            self.client = zhipuai.ZhipuAI(api_key=final_api_key)
            self.model_name = model_name
            self.logger.info(
                f"ZhipuAI 客户端已使用模型 '{self.model_name}' 成功初始化。"
            )
        except Exception as e:
            # 捕获客户端初始化过程中可能发生的各种错误，并提供诊断建议。
            self.logger.error(
                f"Generator 初始化错误: 初始化 ZhipuAI 客户端失败。错误详情: {e}",
                exc_info=True,
            )
            self.logger.error(f"请确认以下几点：")
            self.logger.error(
                f"  - 提供的 API Key 是否有效且具有调用模型 '{self.model_name}' 的权限。"
            )
            self.logger.error(
                f"  - 'zhipuai' Python 库是否已正确安装 (例如，通过 pip install zhipuai)。"
            )
            self.logger.error(f"  - 网络连接是否正常，能否访问 ZhipuAI API 服务端点。")
            raise RuntimeError(f"ZhipuAI客户端初始化失败: {e}") from e

        self.logger.info(
            "Generator 初始化成功完成。我已经准备好与大模型交互，生成答案了。"
        )

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        根据用户提供的原始查询和由 Retriever 返回的文档上下文列表，调用大语言模型 (LLM) 生成回答。
        这是RAG流程的最后一个关键步骤。

        Args:
            query (str): 用户提出的原始问题或查询字符串。
            context (List[Dict[str, Any]]): 一个文档字典列表，通常由 Retriever 的 `retrieve` 方法返回。
                                 每个字典代表一个检索到的相关文档，应包含诸如 'id', 'text',
                                 'image_path', 'score' 等信息。这些信息将作为LLM生成答案的依据。

        Returns:
            str: 由大语言模型生成并经过基本后处理的文本响应。
                 如果在调用 LLM API 时发生错误，会返回一条包含错误信息的提示性字符串。
        """
        self.logger.info(f"开始为查询生成最终响应...")
        self.logger.info(
            f"  接收到的用户查询: '{query[:100]}{'...' if len(query)>100 else ''}'"
        )
        self.logger.info(f"  使用 {len(context)} 个检索到的文档作为生成上下文。")

        # --- 步骤 1: 构建发送给 LLM 的 Prompt (通常表现为消息列表 `messages`) ---
        # Prompt的质量直接影响LLM的输出。我必须仔细构建系统指令和格式化上下文。
        self.logger.debug(
            "  - Generator步骤 1: 构建 Prompt (包含系统指令、上下文和用户查询)..."
        )
        messages_for_llm: List[Dict[str, str]] = []
        try:
            messages_for_llm = self._build_messages(query, context)

            if messages_for_llm:
                # 为了日志清晰，打印系统指令和用户查询的部分内容。
                if messages_for_llm[0]["role"] == "system":
                    system_prompt_content = messages_for_llm[0]["content"]
                    context_start_marker = "# 参考文档:"
                    context_start_index = system_prompt_content.find(
                        context_start_marker
                    )
                    if context_start_index != -1:
                        system_instructions_part = system_prompt_content[
                            :context_start_index
                        ].strip()
                        self.logger.debug(
                            f"    生成的系统消息 (指令部分): {system_instructions_part[:400]}{'...' if len(system_instructions_part)>400 else ''}"
                        )
                    else:  # Fallback if marker not found (unlikely with current prompt)
                        self.logger.debug(
                            f"    生成的系统消息 (部分): {system_prompt_content[:400]}{'...' if len(system_prompt_content)>400 else ''}"
                        )

                if len(messages_for_llm) > 1 and messages_for_llm[1]["role"] == "user":
                    self.logger.debug(
                        f"    生成的用户消息 (原始查询): {messages_for_llm[1]['content']}"
                    )
            self.logger.debug("    Prompt 构建完成。")
        except Exception as e_build_prompt:
            self.logger.error(
                f"错误: 构建 Prompt 时发生异常: {e_build_prompt}", exc_info=True
            )
            return "抱歉，在准备向语言模型发送请求时遇到了内部错误（Prompt构建失败）。"

        # --- 步骤 2: 调用 ZhipuAI Chat Completions API ---
        # 这是与外部API交互的关键步骤，必须处理各种可能的API错误。
        self.logger.info(
            f"  - Generator步骤 2: 开始调用 ZhipuAI Chat API (使用模型: {self.model_name})..."
        )
        llm_raw_response_content = (
            "抱歉，在尝试从语言模型生成响应时遇到了一个未知问题。"  # 默认错误消息
        )
        try:
            # 确保 messages_for_llm 是有效的列表
            if not isinstance(messages_for_llm, list) or not messages_for_llm:
                self.logger.error(
                    "错误：构建的 Prompt 消息列表无效或为空，无法调用 LLM API。"
                )
                return "抱歉，在准备向语言模型发送请求时遇到了内部错误（Prompt为空）。"

            # 调用 API，传入模型名称、消息列表、温度和最大Token数。
            # temperature 控制随机性，max_tokens 控制输出长度。
            api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_llm,
                temperature=0.7,  # 一个常用的值，平衡创造性和事实性
                max_tokens=1500,  # 控制生成的最长响应
                # stream=False # 确保不是流式输出，等待完整响应
            )

            # 检查API响应的结构，提取生成的文本内容。
            # ZhipuAI SDK v2 的响应结构可能略有不同，需要适配
            if api_response and api_response.choices and len(api_response.choices) > 0:
                choice = api_response.choices[0]
                if choice.message and choice.message.content:
                    llm_raw_response_content = choice.message.content
                    self.logger.info(f"    ZhipuAI API 调用成功。已接收到模型的响应。")
                else:
                    self.logger.warning(
                        "    ZhipuAI API 调用似乎成功，但响应中的 message 或 content 为空。将使用默认错误消息。"
                    )
                    self.logger.debug(
                        f"    实际的 API choice 对象: {choice.model_dump_json(indent=2)}"
                    )  # Use model_dump_json for Pydantic models
            else:
                self.logger.warning(
                    "    ZhipuAI API 调用似乎成功，但响应结构不符合预期 (choices 列表为空或不存在)。将使用默认错误消息。"
                )
                # Log the actual response for debugging if it's not as expected
                if api_response:
                    self.logger.debug(
                        f"    实际的 API 响应对象: {api_response.model_dump_json(indent=2)}"
                    )  # Use model_dump_json
                else:
                    self.logger.debug("    实际的 API 响应对象为 None 或 False。")

            # 记录Token使用情况，这对成本控制很重要。
            if hasattr(api_response, "usage") and api_response.usage:
                completion_tokens = getattr(
                    api_response.usage, "completion_tokens", "N/A"
                )
                prompt_tokens = getattr(api_response.usage, "prompt_tokens", "N/A")
                total_tokens = getattr(api_response.usage, "total_tokens", "N/A")
                self.logger.info(
                    f"      Token 使用情况 -> 输入提示: {prompt_tokens} tokens, 生成响应: {completion_tokens} tokens, 总计: {total_tokens} tokens."
                )
            else:
                self.logger.info("      未能从 API 响应中获取详细的 token 使用情况。")

        # 处理不同类型的 ZhipuAI API 调用错误，提供有针对性的错误信息。
        # 注意：错误类型可能随 zhipuai SDK 版本变化，以下是基于常见情况的示例。
        except zhipuai.APIStatusError as e_status:
            self.logger.error(
                f"  错误：ZhipuAI API 返回了状态错误。这通常是由于请求参数、权限或账户问题。"
            )
            self.logger.error(f"        HTTP 状态码: {e_status.status_code}")
            # Try to get more details if available in the response body
            error_body = getattr(e_status, "body", None)
            error_message_detail = str(error_body) if error_body else str(e_status)
            self.logger.error(f"        错误详情: {error_message_detail}")
            llm_raw_response_content = (
                f"抱歉，调用语言模型时遇到 API 错误 (状态码: {e_status.status_code})。 "
                f"请检查您的 API Key、账户状态或请求参数，或稍后重试。错误信息: {error_message_detail[:200]}{'...' if len(error_message_detail)>200 else ''}"
            )  # Limit length
        except zhipuai.APIConnectionError as e_conn:
            self.logger.error(f"  错误：无法连接到 ZhipuAI API 服务器: {e_conn}")
            llm_raw_response_content = (
                "抱歉，无法连接到语言模型服务。 "
                "请检查您的网络连接，或确认 ZhipuAI API 端点是否正确且可访问。"
            )
        except (
            zhipuai.APIRequestFailedError
        ) as e_req_failed:  # Catching a potentially relevant error type
            self.logger.error(f"  错误: ZhipuAI API 请求失败: {e_req_failed}")
            error_message_detail = str(e_req_failed)  # Get the string representation
            llm_raw_response_content = f"抱歉，语言模型API请求失败。可能原因包括请求参数无效或服务内部错误。详情: {error_message_detail[:200]}{'...' if len(error_message_detail)>200 else ''}"  # Limit length
        except zhipuai.APITimeoutError as e_timeout:
            self.logger.error(f"  错误: ZhipuAI API 请求超时: {e_timeout}")
            llm_raw_response_content = (
                "抱歉，与语言模型的通信超时。请稍后重试，或检查网络延迟。"
            )
        except Exception as e_unknown:
            self.logger.error(
                f"  错误：调用 LLM 时发生未预料的异常: {e_unknown}", exc_info=True
            )
            llm_raw_response_content = (
                "抱歉，在与语言模型交互并生成响应的过程中，发生了一个意外的内部错误。 "
                "请查看详细日志以获取更多信息。"
            )

        # --- 步骤 3: 对 LLM 的原始响应进行后处理 ---
        # 对原始响应进行清理，使其更适合最终展示。
        self.logger.debug("  - Generator步骤 3: 对 LLM 的原始响应进行后处理...")
        final_processed_response = self._postprocess_response(llm_raw_response_content)
        self.logger.debug("    LLM 响应后处理完成。")

        self.logger.info("LLM 响应生成流程结束。")
        return final_processed_response

    def _build_messages(
        self, query: str, context: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        根据用户查询和检索到的上下文文档，构建符合 ZhipuAI Chat API 要求的消息列表 (messages)。
        这个列表通常包含两条主要消息：
        1.  **System Message (`role: "system"`)**: 提供系统级的指令，设定 LLM 的角色、行为规则，
            并在此处注入检索到的上下文信息（格式化为文本）。
        2.  **User Message (`role: "user"`)**: 包含用户原始的查询字符串。

        构建高质量的Prompt是引导LLM输出正确答案的关键，我必须仔细设计这里的指令。

        Args:
            query (str): 用户的原始查询。
            context (List[Dict[str, Any]]): Retriever 返回的上下文文档列表。每个文档字典应包含 'id', 'text',
                                 'image_path', 'score' 等键。

        Returns:
            List[Dict[str, str]]: 一个包含字典的列表，每个字典有 'role' 和 'content' 键，
                                  可以直接传递给 ZhipuAI Chat API 的 `messages` 参数。
                                  如果发生内部错误，可能返回空列表。
        """
        self.logger.debug("开始构建用于 LLM 的消息列表...")
        # 定义系统消息的内容，包含详细的角色设定和行为约束。
        system_message_content_parts = [
            '你是一个高度专业且严谨的文档问答助手。你的任务是根据下面提供的 "参考文档" 部分中的信息来精确地回答用户提出的问题。',
            "\n# 核心指令与行为准则:",
            '1.  **严格依据参考信息**: 你的回答必须 **完全且仅** 基于 "参考文档" 中明确提供的信息。严禁使用任何你在训练数据中学习到的外部知识、个人观点、进行任何形式的推断、猜测或联想超出文档内容。这是确保回答可靠性的最重要原则。',
            '2.  **处理信息不足**: 如果 "参考文档" 中的信息不足以回答用户的问题，或者问题与所有提供的文档内容均不相关，你必须明确指出信息的缺乏。标准回答是：“根据提供的参考文档，我无法找到回答该问题所需的信息。”或者类似表述，如“参考文档中没有包含足够的信息来回答关于...的问题。”。不要试图编造答案，诚实地报告信息不足是专业的表现。',
            '3.  **关于图像内容的理解**: 你无法直接“看到”或解析图像文件本身的内容。你对图像的理解 **必须且只能** 来源于 "参考文档" 中与该图像关联的 **文本描述内容**，以及文档中可能提及的 **图像文件名**。绝不能声称你能直接感知图像内容或对其进行视觉分析。',
            "4.  **回答涉及图像的问题**:",
            '    - 如果用户的问题涉及到某张图片（例如，通过图片文件名或描述性提问），请首先在 "参考文档" 的文本描述中仔细查找是否有与该图片相关的说明。',
            "    - 如果找到了相关的文本描述，请依据该文本描述来回答。",
            "    - 如果文档中只提供了图片的文件名但没有相应的文本描述，你可以提及这个文件名（例如，“文档提到了一个名为 'circuit_diagram.png' 的关联图片”），并明确说明文档中缺少对该图片内容的具体文字描述，因此无法进一步回答。",
            "    - 如果文档中既没有图片描述也没有文件名信息，或者问题与文档中提及的任何图片都无关，请按照上述第2条“处理信息不足”的规则进行回复。",
            "5.  **引用来源 (推荐)**: 在可能的情况下，如果你的答案基于某一个或某几个特定的参考文档，请在回答中指明这些来源。例如：“根据文档 ID 'BGREF_01' 的描述...” 或 “参考文档 1 (ID: XXX) 和文档 3 (ID: YYY) 提到...”。这有助于用户追溯信息源，提高答案的可信度。",
            "6.  **回答风格与格式**: 你的回答应尽可能地简洁、清晰、直接，并且专业。避免使用冗长的前缀、不必要的客套话或模棱两可的表述。如果答案包含多个要点，可以使用列表或分点来组织，以提高可读性。",
            "\n# 参考文档:",
            "--- 开始参考文档部分 ---",
        ]
        system_message_content = "\n".join(system_message_content_parts).strip()

        context_parts_for_prompt: List[str] = []
        # 检查是否有检索到的上下文。如果没有，需要告知LLM。
        if not context:
            self.logger.info(
                "    注意: 未向LLM提供任何检索到的上下文文档 (可能是因为检索无结果)。"
            )
            context_parts_for_prompt.append(
                "\n（系统提示：本次未能从知识库中检索到与用户问题相关的文档。请基于此情况进行回答，并遵循“处理信息不足”的规则。）"
            )
        else:
            # 遍历每个检索到的文档，将其格式化为易于LLM理解的文本块。
            self.logger.info(
                f"    正在将 {len(context)} 个检索到的文档格式化为 LLM 的上下文..."
            )
            for i, doc_info in enumerate(context):
                doc_id = doc_info.get("id", "未知ID")  # 获取文档ID，提供默认值。
                score_value = doc_info.get("score", "N/A")  # 获取相关度得分。
                text_content = doc_info.get(
                    "text", "无可用文本内容"
                )  # 获取文本内容，提供默认值。
                image_file_path = doc_info.get("image_path")  # 获取图像路径。

                # 格式化图像信息，如果存在图像路径的话。
                image_filename = (
                    os.path.basename(image_file_path) if image_file_path else None
                )
                image_info_str = (
                    f"关联图片文件名: '{image_filename}'"
                    if image_filename
                    else "无明确关联的图片信息"
                )

                # 截断过长的文本内容，避免Prompt超出LLM的上下文窗口限制。
                # 需要估算Token长度而不是字符长度，但简单截断作为近似。
                max_text_len_for_llm = 700  # Character limit (approximate token limit)
                truncated_text_content = text_content[:max_text_len_for_llm] + (
                    "..." if len(text_content) > max_text_len_for_llm else ""
                )

                # 构建单个文档的格式化字符串。
                doc_context_parts = [
                    f"\n--- 参考文档 {i+1} ---",  # 添加文档分隔符和编号。
                    f"  原始文档ID: {doc_id}",  # 添加原始ID。
                    (
                        f"  与查询的相关度得分: {score_value:.4f}"
                        if isinstance(score_value, float)
                        else f"  与查询的相关度得分: {score_value}"
                    ),  # 添加相关度得分，如果是浮点数则格式化。
                    f"  文本内容摘要: {truncated_text_content}",  # 添加文本内容摘要。
                    f"  {image_info_str}",  # 添加图像信息。
                ]
                context_parts_for_prompt.extend(doc_context_parts)

        # 将所有文档的格式化字符串合并。
        formatted_context_section = "\n".join(context_parts_for_prompt)
        # 将格式化后的上下文添加到系统消息的末尾。
        system_message_content += (
            "\n" + formatted_context_section + "\n--- 结束参考文档部分 ---"
        )

        # 构建最终的消息列表，包括系统消息和用户消息。
        final_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": query},
        ]
        self.logger.debug(
            f"为 LLM 构建的消息列表完成。共 {len(final_messages)} 条消息。"
        )
        # 增加一个简单的验证，确保内容不是空的
        if not system_message_content.strip() or not query.strip():
            self.logger.error("错误：构建的系统消息或用户查询内容为空！")
            return []  # Return empty list to indicate failure

        return final_messages

    def _postprocess_response(self, llm_raw_response: str) -> str:
        """
        对从 LLM API 获取的原始响应字符串进行基本的后处理。
        目前主要执行去除首尾空白字符的操作。
        未来可以根据需要在这里添加更复杂的处理逻辑，例如移除特定的模型习语、修正格式等。

        Args:
            llm_raw_response (str): 从 LLM API 收到的原始文本响应。

        Returns:
            str: 经过后处理的文本响应，准备好呈现给用户或用于后续流程。
        """
        self.logger.debug(
            f"开始对 LLM 原始响应进行后处理。原始响应 (前100字符): '{llm_raw_response[:100]}...'"
        )
        # 主要执行去除首尾空白
        processed_response = llm_raw_response.strip()

        # 可以在这里添加更多后处理逻辑，例如：
        # 1. 移除模型可能添加的冗余前缀或后缀。
        #    Example:
        #    prefixes_to_remove = ["好的，根据您提供的文档，", "根据参考文档："]
        #    for prefix in prefixes_to_remove:
        #        if processed_response.startswith(prefix):
        #            processed_response = processed_response[len(prefix):].strip()
        #            self.logger.debug(f"  移除了前缀 '{prefix}'。")
        #            break # Assuming only one prefix needs removal
        #
        # 2. 格式修正（例如，确保列表格式正确）。
        # 3. 敏感信息过滤（如果需要）。

        self.logger.debug(
            f"LLM 响应后处理完成。处理后响应 (前100字符): '{processed_response[:100]}...'"
        )
        return processed_response

    def close(self):
        """
        关闭 Generator 实例时调用的清理方法。
        ZhipuAI 客户端通常不需要显式关闭。此方法主要用于记录Generator的生命周期结束。
        """
        self.logger.info("开始关闭 Generator 实例...")
        # ZhipuAI 客户端通常由其内部管理连接，无需显式关闭。
        # 如果未来需要管理特定资源（如文件句柄），应在此处添加关闭逻辑。
        self.logger.info("Generator 实例关闭完成。")
