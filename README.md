# ScholarFlow
这是一个非常系统且庞大的工程。为了让你能够从0到1真正落地，我将这个项目命名为 **“ScholarFlow (智流)”**，并将开发过程拆分为 **五个阶段**。我们将采用 **前后端分离** 的架构，以保证扩展性和演示效果。

---

### 📂 项目概览：ScholarFlow (智流)
*   **核心理念：** 学习资料多模态转化 + 智能代理规划
*   **开发难度：** ⭐⭐⭐⭐⭐ (建议按模块开发，不要试图一次写完)
*   **预计开发周期：** 个人全职开发约 4-6 周

---
scholar-flow/
├── .github/                   # GitHub Actions CI/CD 配置 (进阶)
│   └── workflows/
│       └── python-app.yml     # 自动运行测试
├── backend/                   # 后端工程 (FastAPI + AI Agents)
│   ├── app/
│   │   ├── api/               # API 路由层
│   │   │   ├── __init__.py
│   │   │   ├── endpoints.py   # 具体的接口定义
│   │   │   └── dependencies.py # 依赖注入
│   │   ├── core/              # 核心配置
│   │   │   ├── config.py      # 读取 .env 配置
│   │   │   └── security.py    # 如果有登录功能
│   │   ├── services/          # 核心业务逻辑 (AI 的大脑在这里)
│   │   │   ├── __init__.py
│   │   │   ├── llm_engine.py  # 封装 OpenAI/DeepSeek 调用
│   │   │   ├── podcast.py     # 播客生成逻辑
│   │   │   ├── video_rag.py   # 视频检索逻辑
│   │   │   ├── graph.py       # 知识图谱构建
│   │   │   └── agent.py       # 学习计划 Agent
│   │   ├── models/            # Pydantic 数据模型定义
│   │   │   └── schemas.py     # Request/Response 格式
│   │   ├── utils/             # 工具函数
│   │   │   ├── file_parser.py # PDF/SRT 解析
│   │   │   └── audio.py       # 音频拼接工具
│   │   └── main.py            # FastAPI 入口文件
│   ├── tests/                 # 单元测试
│   ├── data/                  # 本地存储 (会被 .gitignore 忽略)
│   │   ├── chromadb/          # 向量数据库文件
│   │   └── temp/              # 临时生成的 MP3/MP4
│   ├── Dockerfile             # 后端镜像构建
│   ├── requirements.txt       # Python 依赖
│   └── .env.example           # 环境变量模板
├── frontend/                  # 前端工程 (Next.js + React)
│   ├── public/                # 静态资源 (Logo, Icons)
│   ├── src/
│   │   ├── app/               # Next.js App Router 页面
│   │   │   ├── page.tsx       # 首页
│   │   │   ├── podcast/       # 播客生成页
│   │   │   └── graph/         # 知识图谱页
│   │   ├── components/        # UI 组件 (Shadcn/ui)
│   │   │   ├── ui/            # 基础按钮、输入框
│   │   │   ├── GraphView.tsx  # 3D 图谱组件
│   │   │   └── VideoPlayer.tsx
│   │   ├── lib/               # 工具函数 (API Client)
│   │   │   ├── api.ts         # Axios/Fetch 封装
│   │   │   └── utils.ts
│   │   └── types/             # TypeScript 类型定义
│   ├── Dockerfile
│   ├── package.json
│   ├── tailwind.config.ts
│   └── next.config.mjs
├── docs/                      # 项目文档
│   ├── architecture.png       # 架构图
│   └── api_docs.md            # 接口文档
├── .gitignore                 # 忽略 node_modules, venv, .env, __pycache__
├── docker-compose.yml         # 一键启动前后端
├── LICENSE                    # MIT License
└── README.md                  # 项目主页说明书


### 🛠️ 第一部分：技术栈选型 (2025 主流配置)

#### 1. 后端 (Backend) - “大脑”
*   **框架：** **FastAPI** (Python)。高性能，天生支持异步（处理视频/大模型请求必须异步）。
*   **LLM 接入：** **LangChain** + **OpenAI API (GPT-4o)** 或 **DeepSeek API** (性价比高)。
*   **向量数据库：** **ChromaDB** (轻量级，本地运行，无需服务器)。
*   **图数据库：** **NetworkX** (初期开发用) -> **Neo4j** (进阶用)。
*   **多媒体处理：**
    *   视频/音频：`FFmpeg` (核心工具), `OpenAI Whisper` (语音转文字).
    *   语音合成：`edge-tts` (微软免费接口，效果极佳) 或 `CosyVoice`.
    *   PDF解析：`PyMuPDF`.

#### 2. 前端 (Frontend) - “颜值”
*   **框架：** **Next.js (React)**。生态最强，方便做复杂的交互。
*   **UI 组件库：** **Shadcn/ui** + **Tailwind CSS** (现代、美观、开发快)。
*   **可视化库：**
    *   知识图谱：`react-force-graph-3d` (实现炫酷的3D节点拖拽)。
    *   日程表：`react-big-calendar`。

---

### 🚀 第二部分：从 0 到 1 开发教程

#### 环境准备
创建项目文件夹并初始化：
```bash
mkdir scholar-flow
cd scholar-flow
mkdir backend frontend
# 后端依赖安装
cd backend
python -m venv venv
source venv/bin/activate  # Mac/Linux
# .\venv\Scripts\activate # Windows
pip install fastapi uvicorn langchain openai chromadb pymupdf edge-tts networkx moviepy openai-whisper
```

---

#### 阶段一：实现“一键生成播客” (The Podcast Engine)
这是最容易上手且效果最震撼的功能。

**1. 后端逻辑 (`backend/podcast.py`)**
核心思路：PDF -> 文本 -> GPT写剧本 -> TTS转语音 -> 拼接。

```python
import edge_tts
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import fitz  # PyMuPDF

# 1. 提取PDF文本
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text[:20000] # 限制长度防止Token溢出

# 2. 生成对话脚本
def generate_script(text):
    llm = ChatOpenAI(model="gpt-4o", api_key="YOUR_KEY")
    prompt = ChatPromptTemplate.from_template(
        """
        你是一个科普广播剧编剧。将以下学习资料改编成一段双人对话（Host A 和 Host B）。
        Host A: 好奇心强，负责提问和捧哏。
        Host B: 专家人设，负责幽默地解释核心概念。
        请直接输出JSON格式列表，例如：
        [{"speaker": "A", "text": "哇，这个概念听起来好难啊"}, {"speaker": "B", "text": "其实很简单..."}]
        
        资料内容：{text}
        """
    )
    chain = prompt | llm
    return chain.invoke({"text": text}).content

# 3. 语音合成 (核心)
async def synthesize_audio(script_json):
    # 解析JSON (此处省略json.loads)
    script = script_json 
    final_audio = []
    
    for line in script:
        # A用男声，B用女声
        voice = "en-US-GuyNeural" if line['speaker'] == "A" else "en-US-JennyNeural"
        # 中文可用 "zh-CN-YunxiNeural" 和 "zh-CN-XiaoxiaoNeural"
        communicate = edge_tts.Communicate(line['text'], voice)
        filename = f"temp_{line['index']}.mp3"
        await communicate.save(filename)
        # 这里需要用 pydub 或 moviepy 将所有 temp mp3 拼接到一起
        # ...拼接逻辑省略...
    
    return "final_podcast.mp3"
```

---

#### 阶段二：实现“视频笔记索引” (Video RAG)
核心思路：Whisper 识别字幕 -> 存入向量库 -> 语义搜索 -> FFmpeg 截取。

**1. 建立向量库 (`backend/vector_store.py`)**

```python
import chromadb
from chromadb.utils import embedding_functions

# 初始化本地向量库
client = chromadb.PersistentClient(path="./db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="video_subs", embedding_function=sentence_transformer_ef)

def index_video_captions(video_id, subtitles):
    # subtitles 是一个列表: [{'start': 10, 'end': 15, 'text': '二叉树反转...'}]
    ids = [f"{video_id}_{i}" for i in range(len(subtitles))]
    documents = [sub['text'] for sub in subtitles]
    metadatas = [{"start": sub['start'], "end": sub['end'], "vid": video_id} for sub in subtitles]
    
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

def search_content(query):
    results = collection.query(query_texts=[query], n_results=1)
    # 返回最佳匹配的时间戳
    best_match = results['metadatas'][0][0]
    return best_match['start'], best_match['end']
```

**2. 视频截取 (`backend/video_processor.py`)**
使用 `moviepy` 库根据上面查到的 `start` 和 `end` 时间截取视频。

```python
from moviepy.editor import VideoFileClip

def clip_video(video_path, start, end, output_path):
    with VideoFileClip(video_path) as video:
        new = video.subclip(start, end)
        new.write_videofile(output_path, audio_codec='aac')
```

---

#### 阶段三：实现“3D 知识图谱” (The Knowledge Graph)
核心思路：LLM 提取三元组 -> 前端 3D 渲染。

**1. 实体抽取 Prompt**
```python
prompt = """
分析这段文本，提取所有的核心概念实体以及它们之间的关系。
输出格式为 JSON 列表: [{"source": "线粒体", "target": "ATP", "relationship": "合成"}, ...]
文本: {text}
"""
```

**2. 前端可视化 (React)**
在 `frontend` 目录下安装依赖：
```bash
npm install react-force-graph-3d
```

**React 组件代码 (`frontend/components/KnowledgeGraph.js`)**
```jsx
import ForceGraph3D from 'react-force-graph-3d';

const myData = {
  nodes: [ 
    { id: '线粒体', group: 1, status: 'learned' }, 
    { id: 'ATP', group: 2, status: 'unlearned' } 
  ],
  links: [ 
    { source: '线粒体', target: 'ATP', value: '合成' } 
  ]
};

export default function KnowledgeGraph() {
  return (
    <ForceGraph3D
      graphData={myData}
      nodeColor={node => node.status === 'learned' ? 'green' : 'grey'}
      linkLabel={link => link.value}
      onNodeClick={node => alert(`正在为你导航到知识点: ${node.id}`)}
    />
  );
}
```

---

#### 阶段四：实现“Agent 学习管家” (The Planner)
核心思路：自然语言理解 -> 任务拆解 -> 模拟日程 API。

**1. 任务拆解 Agent (`backend/agent.py`)**

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 定义输出结构
class Task(BaseModel):
    task_name: str = Field(description="具体的学习任务")
    duration_minutes: int = Field(description="预计耗时")
    deadline_day: int = Field(description="第几天完成，1代表第一天")

class Plan(BaseModel):
    tasks: List[Task]

def create_study_plan(goal, days):
    parser = PydanticOutputParser(pydantic_object=Plan)
    prompt = ChatPromptTemplate.from_template(
        "请将目标 '{goal}' 拆解为 {days} 天的学习计划。\n{format_instructions}"
    )
    # ... Invoke LLM ...
    return parser.parse(llm_output)
```

**2. 动态调整逻辑**
这是一个简单的算法逻辑：
*   **输入：** 原计划 JSON，用户未完成的任务 ID。
*   **处理：** 遍历 JSON，找到未完成任务，将其 `deadline_day` + 1。
*   **检查：** 如果某一天总时长 > 8小时，提示“超负荷”，询问是否延期整个计划。

---

### 🎨 整合 API 接口 (FastAPI)

最后，在 `backend/main.py` 中将所有功能暴露为 API：

```python
from fastapi import FastAPI, UploadFile, File
from podcast import generate_podcast_flow
from vector_store import search_content

app = FastAPI()

@app.post("/upload/pdf-to-podcast")
async def create_podcast(file: UploadFile = File(...)):
    # 1. 保存文件
    # 2. 提取文本
    # 3. 生成音频
    return {"audio_url": "/static/final_podcast.mp3"}

@app.get("/query/video")
async def query_video(question: str, video_id: str):
    start, end = search_content(question)
    # 触发剪辑逻辑
    return {"clip_url": f"/static/clips/{video_id}_{start}_{end}.mp4"}
```

---

### 📅 开发路线图 (Roadmap)

1.  **Week 1: 基础建设**
    *   搭建 FastAPI 和 Next.js 框架。
    *   跑通 OpenAI API 调用。
    *   **里程碑：** 实现上传 PDF，前端能显示提取出的纯文本。

2.  **Week 2: 播客模块 (MVP)**
    *   集成 `edge-tts`。
    *   实现“文本 -> 对话脚本 -> 音频”的链路。
    *   **里程碑：** 上传论文，能听到两个 AI 在聊天。

3.  **Week 3: 视频 RAG**
    *   集成 Whisper (如果是本地跑比较慢，建议用 OpenAI 的 Whisper API)。
    *   搭建 ChromaDB。
    *   **里程碑：** 问“Transformer架构在哪里讲的？”，系统返回具体视频切片。

4.  **Week 4: 知识图谱与前端炫技**
    *   实现三元组提取。
    *   集成 `react-force-graph-3d`。
    *   **里程碑：** 屏幕上出现一个可旋转的 3D 知识球体。

5.  **Week 5: Agent 与收尾**
    *   写 Prompt 做任务拆解。
    *   简单的日程表界面。
    *   整体 UI 美化。

### 💡 给开发者的特别建议
1.  **不要一开始就做真数据库：** 所有的用户数据、任务数据先存 JSON 文件，跑通逻辑最重要。
2.  **Token 成本：** 开发阶段用 DeepSeek 或 gpt-3.5-turbo，正式演示再切 gpt-4o，否则钱包受不了。
3.  **展示技巧：** 在答辩或演示时，**“图表即代码”** 和 **“3D图谱”** 是最吸睛的，一定要准备好漂亮的测试数据（比如一张复杂的生物代谢图）。
