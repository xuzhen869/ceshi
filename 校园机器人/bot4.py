import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
import chromadb
import os
import glob


# ========== 1. 自定义 Embedding 函数（使用本地模型） ==========
class LocalEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.__name__ = "LocalEmbeddingFunction"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        return embeddings.tolist()


# ========== 2. 加载文档到向量数据库 ==========
def load_documents_to_db(folder_path="./knowledge_base", db_path="./campus_db"):
    """从文件夹加载文档，存入向量数据库"""

    # 读取所有txt文件
    documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return 0

    file_paths = glob.glob(f"{folder_path}/*.txt")

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按空行切分成段落
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for p in paragraphs:
            if p:
                documents.append({
                    "content": p,
                    "source": os.path.basename(file_path)
                })

    if not documents:
        return 0

    # 初始化向量数据库
    embedding_fn = LocalEmbeddingFunction()
    client = chromadb.PersistentClient(path=db_path)

    # 删除旧collection（如果存在）
    try:
        client.delete_collection("campus_knowledge")
    except:
        pass

    # 创建新collection
    collection = client.create_collection(
        name="campus_knowledge",
        embedding_function=embedding_fn
    )

    # 添加文档
    ids = [f"doc_{i}" for i in range(len(documents))]
    contents = [d["content"] for d in documents]
    metadatas = [{"source": d["source"]} for d in documents]

    # 分批添加
    batch_size = 100
    for i in range(0, len(contents), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_contents = contents[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        collection.add(
            documents=batch_contents,
            ids=batch_ids,
            metadatas=batch_metadatas
        )

    return len(documents)


# ========== 3. 校园问答机器人 ==========
class CampusBot:
    def __init__(self, api_key, db_path="./campus_db"):
        self.embedding_fn = LocalEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=db_path)

        # 获取或创建collection
        try:
            self.collection = self.client.get_collection("campus_knowledge")
        except:
            self.collection = self.client.create_collection(
                name="campus_knowledge",
                embedding_function=self.embedding_fn
            )

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.3
        )

        self.is_ready = self.collection.count() > 0
        # 对话记忆 - 简化配置
        self.memory = ConversationBufferMemory(
            return_messages=True
        )

    def retrieve(self, query, top_k=50):
        """语义检索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        docs = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        # 返回文档和元数据的配对
        return list(zip(docs, metadatas))

    def ask(self, question):
        """带记忆的问答"""
        if not self.is_ready:
            return "? 知识库为空，请先上传文档", []

        # 1. 检索相关知识（现在返回配对）
        docs_with_meta = self.retrieve(question)
        docs = [item[0] for item in docs_with_meta]  # 提取文档内容

        # 2. 获取历史对话
        memory_vars = self.memory.load_memory_variables({})
        history_messages = memory_vars.get("history", [])

        # 格式化历史记录
        history_text = ""
        for msg in history_messages:
            if hasattr(msg, 'content'):
                if hasattr(msg, 'type'):
                    role = "用户" if msg.type == "human" else "助手"
                else:
                    role = "用户" if "Human" in str(type(msg)) else "助手"
                history_text += f"{role}: {msg.content}\\\\n"

        # 3. 构建提示词
        context = "\\\\n\\\\n".join(docs) if docs else "无相关资料"

        system_prompt = f"""你是校园助手。请基于以下资料回答问题。
    如果资料中没有相关信息，请说"根据现有资料无法回答"。
    回答要简洁准确。

    === 参考资料 ===
    {context}

    === 对话历史 ===
    {history_text if history_text else "无历史对话"}

    请回答用户的问题："""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{question}")
        ])

        # 4. 生成答案
        chain = prompt | self.llm
        response = chain.invoke({"question": question})
        answer = response.content

        # 5. 保存到记忆
        self.memory.save_context({"input": question}, {"output": answer})

        # 返回答案和带元数据的文档
        return answer, docs_with_meta

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()


# ========== 4. Streamlit UI ==========
st.set_page_config(page_title="校园问答机器人", page_icon="🎓", layout="wide")

st.title("🎓 校园问答机器人")
st.caption("基于语义搜索的智能问答系统 | 使用 Embedding 模型理解语义")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("DeepSeek API Key", type="password", placeholder="请输入你的API Key")

    st.markdown("---")
    st.header("📁 知识库管理")
    # 在侧边栏中添加（在 st.markdown("---") 后面或其他合适位置）
    st.markdown("---")
    if st.button(" 清空对话历史", use_container_width=True):
        if 'bot' in st.session_state and st.session_state.bot:
            st.session_state.bot.clear_memory()
            st.session_state.messages = []
            st.rerun()
    # 显示知识库状态
    if os.path.exists("./knowledge_base"):
        txt_files = glob.glob("./knowledge_base/*.txt")
        st.info(f"📄 发现 {len(txt_files)} 个txt文件")

    # 初始化/更新知识库按钮
    if st.button("🔨 初始化知识库", type="primary"):
        with st.spinner("正在加载文档并构建向量数据库..."):
            count = load_documents_to_db()
            if count > 0:
                st.success(f"✅ 已加载 {count} 条文档")
                st.rerun()
            else:
                st.warning("⚠️ 没有找到文档，请在 knowledge_base 文件夹下放入txt文件")

    st.markdown("---")
    st.caption("💡 提示：\n1. 在 knowledge_base 文件夹放入txt文件\n2. 点击「初始化知识库」\n3. 输入API Key即可开始问答")

# 检查API Key
if not api_key:
    st.warning("⚠️ 请在左侧输入你的 DeepSeek API Key")
    st.stop()

# 检查并初始化bot（使用会话状态保存）
if "bot" not in st.session_state:
    st.session_state.bot = None
    st.session_state.current_api_key = None

# API key改变时才重新创建bot
if st.session_state.current_api_key != api_key or st.session_state.bot is None:
    try:
        st.session_state.bot = CampusBot(api_key)
        st.session_state.current_api_key = api_key
        st.session_state.messages = []  # 清空旧对话
    except Exception as e:
        st.error(f"初始化失败：{e}")
        st.info("💡 请先点击「初始化知识库」按钮")
        st.stop()

bot = st.session_state.bot

if not bot.is_ready:
    st.info("📖 知识库为空，请先在左侧点击「初始化知识库」")
    with st.expander("📝 查看示例文档格式"):
        st.code("""
学籍管理规定：
每学期开学两周内办理注册手续。未按时注册按自动退学处理。

课程考核：
百分制60分及格，不及格可以补考一次。

选课流程：
每学期第16-18周进行选课，分三轮进行。
        """, language="text")

    # 提供示例文档下载
    example_content = """学籍管理规定：
每学期开学两周内办理注册手续。未按时注册按自动退学处理。

课程考核：
百分制60分及格，不及格可以补考一次。

选课流程：
每学期第16-18周进行选课，分三轮进行。

毕业要求：
修满160学分，通过毕业论文答辩。
"""
    st.download_button(
        label="📥 下载示例文档",
        data=example_content,
        file_name="example.txt",
        mime="text/plain"
    )
    st.stop()

# 快捷问题
st.subheader("🔍 试试这些问题")
cols = st.columns(4)
quick_questions = ["怎么选课？", "挂科了怎么办？", "奖学金怎么评？", "怎么请假？"]
for i, q in enumerate(quick_questions):
    if cols[i].button(q, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": q})
        st.rerun()

# 聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考..."):
            answer, sources = bot.ask(prompt)
            st.markdown(answer)
            if sources:
                with st.expander("📖 查看参考资料"):
                    for i, (doc, source) in enumerate(sources):
                        st.text(f"[{i + 1}] {doc[:150]}...")
                        if source:
                            st.caption(f"来源: {source.get('source', '未知')}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })