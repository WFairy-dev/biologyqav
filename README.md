本项目是一个基于 RAG 与 大语言模型 的生物学问答系统，支持本地知识库构建、向量检索、智能问答以及可扩展的模型部署方式。
系统采用 Xinference 部署嵌入模型与对话模型，实现灵活可复用的问答体系。

环境配置
```bash
pip install -r requirements.txt
```

初始化数据库

- 设置数据库存储路径：

​			\# on linux or macos

​			export CHATCHAT_ROOT=/path/to/chatchat_data(这个地方换成你的路径)

- 参数配置文件生成：python cli.py init

  知识库文件应放置在：test_data/dataset/data/knowledge_base/{知识库名称}/content/

  这里content里面的内容就可以替换成自己的文件了。

- 运行xinference部署嵌入和对话模型：CUDA_VISIBLE_DEVICES=1 xinference-local --host 0.0.0.0 --port 9998（可参考xinference官方说明文档：https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html）

上述操作完成之后，便可运行python init_database.py -r进行数据库初始化。


启动项目

python startup.py -a

测试数据集可通过链接进行下载：https://doi.org/10.5281/zenodo.17599820

🔗 本系统基于开源项目 Langchain-Chatchat(https://github.com/chatchat-space/Langchain-Chatchat)进行改进与扩展,在此对原作者表示诚挚感谢。


