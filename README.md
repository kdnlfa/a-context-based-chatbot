# 🦜🔗 智能问答机器人

一个基于 LangChain 和智谱AI构建的上下文感知问答机器人，支持文档检索和对话历史记忆功能。

## ✨ 功能特性

- **智能问答**: 基于智谱AI GLM-4-Plus模型的高质量对话体验
- **上下文检索**: 集成 Chroma 向量数据库，支持文档语义检索
- **对话记忆**: 自动维护对话历史，提供连贯的多轮对话体验
- **流式输出**: 实时流式响应，提升用户交互体验
- **Web界面**: 基于 Streamlit 的友好用户界面

## 🛠️ 技术架构

- **前端**: Streamlit Web 应用
- **LLM**: 智谱AI GLM-4-Plus 语言模型
- **向量数据库**: Chroma
- **嵌入模型**: 智谱AI Embedding-3
- **框架**: LangChain

## 📋 环境要求

- Python 3.8+
- 智谱AI API Key

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd a-context-bot
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 环境配置

在项目根目录创建 `.env` 文件，添加您的智谱AI API密钥：

```
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

### 4. 准备向量数据库

确保在项目目录下创建向量数据库目录结构：

```
data_base/
└── vector_db/
    └── chroma/
```

### 5. 运行应用

```powershell
streamlit run streamlit_app.py
```

应用将在浏览器中自动打开，默认地址为 `http://localhost:8501`

## 📁 项目结构

```
a-context-bot/
├── streamlit_app.py      # 主应用文件
├── requirements.txt      # Python依赖包
├── README.md            # 项目说明文档
├── .env                 # 环境变量配置
└── data_base/           # 数据库目录
    └── vector_db/
        └── chroma/      # Chroma向量数据库
```

## 🔧 核心组件

### ZhipuAIEmbeddings
自定义的智谱AI嵌入模型实现，支持：
- 批量文档嵌入（最大64个文档/批）
- 单个查询嵌入
- 自动API调用管理

### ZhipuaiLLM
自定义的智谱AI聊天模型实现，支持：
- 同步和流式文本生成
- 完整的使用统计和响应元数据
- 可配置的模型参数（温度、最大token等）

### 检索增强生成(RAG)
- 智能问题压缩和重写
- 基于对话历史的上下文检索
- 动态文档组合和答案生成

## ⚙️ 配置选项

在 `streamlit_app.py` 中可以调整以下参数：

- **模型名称**: `model_name="glm-4-plus"`
- **温度参数**: `temperature=0.1`
- **向量数据库路径**: `persist_directory='./data_base/vector_db/chroma'`
- **嵌入模型**: `model="embedding-3"`

## 🤝 使用说明

1. **启动应用**: 运行命令后，应用会在浏览器中打开
2. **开始对话**: 在页面底部的输入框中输入您的问题
3. **查看回答**: AI会基于检索到的相关文档内容进行回答
4. **连续对话**: 系统会自动记忆对话历史，支持多轮对话

## ⚠️ 注意事项

- 确保智谱AI API密钥有效且有足够的调用额度
- 首次运行时需要确保向量数据库目录存在
- 建议在生产环境中使用HTTPS
- 定期备份向量数据库数据
