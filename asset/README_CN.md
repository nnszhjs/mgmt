# Asset (静态资源)

本目录包含 RecBole 项目所使用的静态资源，包括 Logo、图片、超参数调优配置文件以及基准测试时间结果。

## 文件

| 文件 / 目录 | 描述 |
|---|---|
| `logo.png` | RecBole 项目 Logo 图片 |
| `framework.png` | RecBole 框架架构图 |
| `new.gif` | 动态 "new" 标识图标 |
| `dataset_list.json` | 列出所有支持的数据集及其元数据的 JSON 文件 |
| `model_list.json` | 列出所有支持的推荐模型及其元数据的 JSON 文件 |
| `questionnaire.xlsx` | 用户反馈或调查问卷表格 |
| `hyper_tune_configs/` | 按推荐类型分类的超参数调优配置文件 |
| `time_test_result/` | 不同推荐模型类别的基准测试时间结果 |

## 子目录

### `hyper_tune_configs/`

包含针对不同数据集和推荐场景的最优超参数配置文件（Markdown 格式），按子目录组织：

| 子目录 | 描述 | 文件 |
|---|---|---|
| `context/` | 上下文感知推荐配置 | `avazu-2m_context.md`、`criteo-4m_context.md`、`ml-1m_context.md` |
| `general/` | 通用推荐配置 | `amazon-books_general.md`、`ml-1m_general.md`、`yelp_general.md` |
| `knowledge/` | 基于知识图谱的推荐配置 | `amazon-books_kg.md`、`lastfm-track_kg.md`、`ml-1m_kg.md` |
| `sequential/` | 序列推荐配置 | `amazon-books_seq.md`、`ml-1m_seq.md`、`yelp_seq.md` |

### `time_test_result/`

包含不同推荐模型类别的基准测试时间结果（Markdown 格式）：

| 文件 | 描述 |
|---|---|
| `General_recommendation.md` | 通用推荐模型的时间基准测试结果 |
| `Sequential_recommendation.md` | 序列推荐模型的时间基准测试结果 |
| `Context-aware_recommendation.md` | 上下文感知推荐模型的时间基准测试结果 |
| `Knowledge-based_recommendation.md` | 基于知识图谱推荐模型的时间基准测试结果 |
