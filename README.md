本项目是一个基于 RAG 与 大语言模型 的生物学问答系统，支持本地知识库构建、向量检索、智能问答以及可扩展的模型部署方式。
系统采用 Xinference 部署嵌入模型与对话模型，实现灵活可复用的问答体系。

## 实验环境（参考）

### 硬件环境

主要硬件配置如下：

| 类别   | 配置说明                                    |
| ---- | --------------------------------------- |
| CPU  | 25 vCPU Intel(R) Xeon(R) Platinum 8481C |
| 内存   | 256GB                                   |
| 显卡   | 2× NVIDIA GeForce RTX 4090              |
| 硬盘   | ≥ 512 GB SSD                            |


### 软件环境

实验软件环境以 Python 为核心，采用 Conda 虚拟环境进行依赖管理，配置如下：

| 类别         | 配置说明         |
| ---------- | ------------ |
| 操作系统       | Ubuntu 22.04 |
| CUDA       | 11.8         |
| Python     | 3.10         |
| ModelScope | 1.20.1       |
| Xinference | 0.16.3       |
| PyTorch    | 2.5.1        |

---

## 环境配置

```bash
pip install -r requirements.txt
```

## 初始化数据库

* 设置数据库存储路径：

  ```bash
  # on linux or macos
  export CHATCHAT_ROOT=/path/to/chatchat_data  # 这里替换成你的实际路径
  ```

* 参数配置文件生成：

  ```bash
  python cli.py init
  ```

* 知识库文件放置路径：

  ```text
  test_data/dataset/data/knowledge_base/{知识库名称}/content/
  ```

  说明：`content/` 目录下的内容可替换为你自己的文件。

* 运行 Xinference 部署嵌入与对话模型：

  ```bash
  CUDA_VISIBLE_DEVICES=1 xinference-local --host 0.0.0.0 --port 9998
  ```

  Xinference 官方安装与启动说明可参考：
  [https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)

* 完成上述步骤后，初始化数据库：

  ```bash
  python init_database.py -r
  ```

## 启动项目

```bash
python startup.py -a
```

## 数据集下载

测试数据集： [https://doi.org/10.5281/zenodo.17599820](https://doi.org/10.5281/zenodo.17599820)
新添加的数据集： [https://doi.org/10.5281/zenodo.18212890](https://doi.org/10.5281/zenodo.18212890)

---

## Tavily API

在需要“外部实时信息检索/联网增强”的场景下，可集成 Tavily 作为外部搜索能力，为 RAG 提供补充证据。

### 安装与配置

```bash
pip install tavily-python
export TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
```

### Python 调用示例

```python
import os
from tavily import TavilyClient

def tavily_search(query: str, k: int = 5):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY. Please export it before running.")
    client = TavilyClient(api_key=api_key)

    resp = client.search(
        query=query,
        search_depth="advanced",  # basic / advanced
        max_results=k,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )

    results = []
    for item in resp.get("results", []):
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
            "score": item.get("score"),
        })
    return results

if __name__ == "__main__":
    q = "Zymomonas mobilis Entner–Doudoroff pathway ethanol fermentation"
    for r in tavily_search(q, k=5):
        print(r["score"], r["title"])
        print(r["url"])
        print((r["content"] or "")[:200], "\n")
```



🔗 本系统基于开源项目 Langchain-Chatchat ([https://github.com/chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)) 进行改进与扩展，在此对原作者表示诚挚感谢。


