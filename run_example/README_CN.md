# Run Example (运行示例)

本目录包含示例脚本和 Jupyter Notebook，用于演示 RecBole 的各种功能。这些示例可作为用户快速上手的实用指南。

## 文件

| 文件 | 描述 |
|---|---|
| `case_study_example.py` | 演示如何在 RecBole 中进行案例研究，包括加载已保存的模型、对特定用户进行全排序 Top-K 预测，以及使用 `full_sort_topk` 和 `full_sort_scores` 工具函数获取物品评分 |
| `save_and_load_example.py` | 演示如何在 RecBole 中保存和加载模型、数据集和数据加载器。展示训练过程中保存检查点，以及使用 `load_data_and_model` 重新加载模型文件、数据集文件和数据加载器文件 |
| `session_based_rec_example.py` | 演示如何使用 RecBole 运行基于会话的推荐基准测试（如 Diginetica、Tmall、NowPlaying），包括自定义配置、数据集分割、模型初始化以及训练/评估工作流 |
| `lstm-model-with-item-infor-fix-missing-last-item.ipynb` | Jupyter Notebook，演示带有物品信息的 LSTM 推荐模型，修复了最后一个物品缺失的问题 |
| `recbole-using-all-items-for-prediction.ipynb` | Jupyter Notebook，演示如何在 RecBole 中使用所有物品进行预测 |
| `sequential-model-fixed-missing-last-item.ipynb` | Jupyter Notebook，演示序列推荐模型，修复了最后一个物品缺失的问题 |
