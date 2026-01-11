# Appendix: Technical Details for Reproducibility

### LLM Model (Standard Generation)
- **Temperature**: 0.9
- **Max Tokens**: 4096
- **History Length**: 10
- **Prompt Name**: `default`
- **Callbacks**: true


### Model Platform Configuration

Models can be loaded from multiple platforms:

 **Xinference** (default):
   - API Base URL: `http://127.0.0.1:9998/v1`
   - Auto-detect models: true






### Embedding Model

**Default Embedding Model**: `bge-m3`

**Configuration**:
- Model: BGE-M3 (multilingual embedding model)
- Dimension: 1024 (for BGE-M3)
- Normalization: L2 normalized



###  Reranker Configuration

**Reranker Model**: `bge-reranker-large`

**Configuration**:
- **Enabled**: true
- **Model**: `bge-reranker-large`
- **Max Input Length**: 1024 tokens
- **Reranker Top-K**: 3 (refinement stage, selects top 3 from initial retrieval)


### Initialization

**Database Initialization**:
```bash
export CHATCHAT_ROOT=/path/to/chatchat_data
python cli.py init
python init_database.py -r  # Rebuild all knowledge bases
```

**Knowledge Base Update**:
```bash
python init_database.py -r -n bioqa  # Rebuild specific KB
python init_database.py -i -n bioqa   # Incremental update
```

### prompt1
  default1: "[Instruction]You are an expert research assistant in synthetic biology
    focusing on Zymomonas mobilis. Using ONLY the information provided in the context,
    reply ONLY with minimal, exact research conclusions, data, or terminology that
    directly answer the question. DO NOT include any additional, inferred, or fabricated
    details and explanation. Your answer must be in the same language as the question.
    `\n\n [Context]: {{context}} \n\n  [Question]: {{question}}\n"
### prompt2
  default4: "[Instruction]: You are a scientific research expert in synthetic biology,
    specializing in Zymomonas mobilis. When answering the multiple-choice question
    below,  Your answer must: • Select exactly only one option (A, B, C, or D) that
    is directly supported by the provided context. • Base your answer strictly on
    the data, research findings, or proper nouns present in the context. • Not add
    any explanation, commentary, or extraneous text. • Ensure your final response
    is exactly one uppercase letter A, B, C, or D with no additional characters, spaces,
    or punctuation. If the context is insufficient, reply exactly: `A`.\n\n[Context]:
    {{context}} \n\n[Question]: {{question}}"

