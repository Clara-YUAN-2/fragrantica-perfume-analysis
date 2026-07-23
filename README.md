# Fragrantica Perfume Analytics & Recommendation System

## 项目简介

本项目基于 Fragrantica 香水数据集，围绕香水市场结构、用户偏好、相似香水推荐和新品香水市场潜力评估展开分析与建模。

项目从数据清洗和探索性分析出发，构建 `Popularity Score` 综合热度指标，识别高热度香水的品牌、国家、Gender 和香调特征；随后基于香水的主香调、前调、中调和后调文本特征，使用 TF-IDF 与余弦相似度构建相似香水推荐系统；最后进一步设计新品香水市场潜力评估模块，模拟品牌方在新品设计阶段基于上市前可知信息判断新品概念是否接近历史高热度香水特征。

本项目最终使用 Streamlit 封装为交互式网页应用，支持用户输入香水名称获取相似香水推荐，也支持输入新品香水概念并输出市场潜力预测结果。

---

## 项目目标

本项目希望回答以下问题：

1. Fragrantica 香水数据中，不同品牌、国家和 Gender 的分布有什么特点？
2. 哪些香调更常见，哪些香调更容易出现在高热度香水中？
3. 仅使用评分是否足以衡量香水受欢迎程度？
4. 如何结合评分和评分人数构造更合理的综合热度指标？
5. 如果用户喜欢某一款香水，能否根据气味结构推荐相似香水？
6. 如果品牌方设计一款新品香水，能否基于历史数据评估其市场潜力？
7. 如何将数据分析和模型结果封装成可交互的数据产品原型？

---

## 数据说明

数据集包含约 2.4 万条香水记录，主要字段包括：

| 字段 | 含义 |
|---|---|
| Perfume | 香水名称 |
| Brand | 品牌 |
| Country | 品牌/香水所属国家 |
| Gender | 香水性别定位，包括 women、men、unisex |
| Rating Value | 用户评分 |
| Rating Count | 评分人数 |
| Year | 发布年份 |
| mainaccord1 - mainaccord5 | 主香调 |
| Top | 前调 |
| Middle | 中调 |
| Base | 后调 |
| Perfumer | 调香师 |

---

## 核心指标设计

### 1. Rating Value

`Rating Value` 表示用户对香水的平均评分，可以反映用户对香水的主观认可程度。

但单独使用评分存在局限：少量用户打出的高分不一定代表广泛认可，因此评分需要结合评分人数一起分析。

### 2. Rating Count

`Rating Count` 表示参与评分的用户数量，可以反映香水受到的关注程度和评价样本规模。

评分人数存在明显长尾分布，大多数香水评分人数较少，少数经典或热门香水拥有大量评分。

### 3. Popularity Score

为了同时考虑评分水平和评分人数，本项目构造综合热度指标：

```text
Popularity Score = Rating Value × log(1 + Rating Count)
```

该指标可以避免只看评分造成误判，也可以避免评分人数过大的香水完全主导结果。

### 4. High Potential

本项目将 `Popularity Score` 排名前 10% 的香水定义为历史高热度香水，并构造二分类标签：

```text
High_Potential = 1 if Popularity Score >= 90% quantile
```

该标签用于新品香水市场潜力预测模型。

---

## 项目结构

```text
fragrantica-perfume-analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
├── fra_cleaned_analysis_ready.csv
│
├── notebooks/
│   ├── perfume_eda_starter.ipynb
│   ├── perfume_advanced_analysis.ipynb
│   ├── perfume_content_based_recommender.ipynb
│   └── perfume_potential_prediction_model.ipynb
│
├── report/
│   └── Fragrantica香水项目_完整报告.docx
│
└── figures/
    └── project_charts_and_screenshots
```

---

## 分析模块

### 1. 数据清洗与特征工程

主要处理内容包括：

- 统一字段名称格式；
- 处理空字符串、unknown、nan 等缺失值；
- 将评分、评分人数和年份转换为数值型；
- 构造 `Popularity Score` 综合热度指标；
- 合并主香调、前调、中调和后调为 `feature_text` / `scent_text` 文本特征。

---

### 2. 探索性数据分析 EDA

本项目从以下维度进行探索性分析：

- Top 20 品牌香水数量分布；
- Top 20 国家香水数量分布；
- Rating Value 评分分布；
- Rating Count 评分人数分布；
- Gender 性别定位分布；
- Top 30 主香调频率；
- 主香调综合热度排名；
- 1990 年以来香水发布趋势。

主要发现包括：

- Fragrantica 数据存在明显品牌集中和国家集中现象；
- 女性香水数量最多，中性香数量也较高；
- 评分整体集中在 3.5–4.5 区间，单独使用评分区分度有限；
- 评分人数呈现明显长尾分布，因此需要结合评分人数构造综合热度指标；
- woody、citrus、aromatic、sweet、fruity 等香调出现频率较高；
- 频率高不一定代表热度高，因此需要进一步分析香调与综合热度之间的关系。

---

### 3. 高热度香水画像分析

本项目将 `Popularity Score` 排名前 10% 的香水定义为高热度香水，并分析其共同特征。

分析内容包括：

- 高热度香水 Top 15；
- 高热度香水中的 Gender 占比差异；
- 高热度香水中的香调 Lift 分析；
- 不同 Gender 的香调差异；
- 品牌数量与品牌平均热度关系。

其中，香调 Lift 指标定义为：

```text
Lift = 高热度组中该香调出现比例 / 整体数据中该香调出现比例
```

Lift 大于 1 表示该香调在高热度香水中被过度代表，可能与用户高关注度存在关联。

---

## 相似香水推荐系统

### 方法说明

本项目构建了基于内容的香水相似推荐系统。

推荐系统使用以下字段构造香水气味文本：

- mainaccord1 - mainaccord5
- Top
- Middle
- Base

随后使用 `TfidfVectorizer` 将气味文本转换为数值向量，并使用余弦相似度计算香水之间的相似程度。

```text
输入一款目标香水
→ 提取其香调与前中后调特征
→ 计算与其他香水的文本相似度
→ 返回最相似的香水列表
```

### 推荐系统特点

- 不依赖用户历史行为数据；
- 推荐结果具有较强可解释性；
- 可以根据气味结构推荐相似香水；
- 设置最低评分人数筛选条件，避免推荐过于冷门或评价样本过少的香水。

---

## 新品香水市场潜力预测模型

### 模型目标

本模块模拟新品香水概念设计场景。

假设品牌方在新品上市前已经知道以下信息：

- Brand
- Country
- Gender
- Year
- 主香调
- 前调
- 中调
- 后调

模型希望基于历史 Fragrantica 数据，判断该新品概念是否更接近历史高热度香水特征。

### 特征选择原则

为了避免数据泄漏，模型只使用上市前可以知道的信息作为输入，不使用 `Rating Count` 作为输入特征。

因为评分人数是香水上市后才会产生的用户行为数据，如果将其作为输入，会造成模型“提前看到未来信息”。

### 模型方法

项目使用：

- Logistic Regression 作为基线模型；
- Random Forest Classifier 作为非线性分类模型；
- Ridge Regression 作为连续热度分数预测模型；
- TF-IDF 处理香调和前中后调文本特征；
- One-Hot Encoding 处理品牌、国家和 Gender 等类别特征。

### 模型输出

新品香水潜力评估模块输出：

- High Potential Probability：进入历史高热度区间的概率；
- Potential Level：潜力等级；
- Prediction Confidence：预测可信度；
- Similar Historical Perfumes：相似历史香水案例。

### 模型边界

本模型并不预测香水是否一定好闻，也不能直接预测真实销量。

香水实际市场表现还会受到以下因素影响：

- 调香水平；
- 原料比例；
- 留香和扩香表现；
- 价格；
- 包装设计；
- 品牌营销；
- 渠道曝光；
- 社交媒体传播。

因此，本模型更适合作为新品概念阶段的市场潜力辅助评估工具。

---

## Streamlit 交互式网页应用

本项目使用 Streamlit 将推荐系统和新品潜力评估模型封装为交互式网页应用。

### 功能 1：Find Similar Perfumes

用户输入香水名或品牌关键词，系统会：

1. 检索匹配香水；
2. 用户选择目标香水；
3. 系统基于 TF-IDF 和余弦相似度返回相似香水推荐结果。

推荐结果展示：

- 香水名称；
- 品牌；
- 国家；
- Gender；
- Rating Value；
- Rating Count；
- Popularity Score；
- Similarity；
- Year。

### 功能 2：New Perfume Potential Evaluator

用户输入新品香水概念，包括：

- Brand；
- Country；
- Gender；
- Year；
- Main accords；
- Top notes；
- Middle notes；
- Base notes。

系统输出：

- 高热度概率；
- 潜力等级；
- 预测可信度；
- 相似历史香水案例；
- 模型解释说明。

---

## 如何运行项目

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Streamlit 应用

```bash
python -m streamlit run app.py
```

或：

```bash
streamlit run app.py
```

运行成功后，终端会显示：

```text
Local URL: http://localhost:8501
```

点击该链接即可打开网页应用。

---

## requirements.txt

项目主要依赖：

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
```

---

## 产品化思考与效果评估方案

如果该推荐系统上线到真实香水内容平台，可以通过 A/B 实验验证推荐效果。

### 实验设计

- 对照组：展示平台热门香水推荐；
- 实验组：展示基于香调相似度的推荐结果。

### 核心评估指标

| 指标 | 含义 |
|---|---|
| CTR | 推荐结果点击率 |
| 收藏率 | 用户是否将推荐香水加入收藏 |
| 详情页停留时长 | 用户是否对推荐内容感兴趣 |
| 愿望单加入率 | 用户是否产生进一步购买/试香意向 |
| 转化率 | 如果平台有购买链路，可观察购买转化 |

如果实验组在点击率、收藏率或愿望单加入率上显著优于对照组，可以说明相似香调推荐比单纯热门推荐更能匹配用户偏好。

---

## 项目亮点

1. 完成从数据清洗、EDA、指标构建到建模和网页应用的完整流程；
2. 构造 `Popularity Score` 综合热度指标，避免单一评分带来的偏差；
3. 使用 Lift 分析识别高热度香水中的过度代表香调；
4. 使用 TF-IDF 和余弦相似度构建可解释的相似香水推荐系统；
5. 构建新品香水市场潜力评估模块，模拟真实业务决策场景；
6. 使用 Streamlit 将分析结果和模型封装为交互式数据产品；
7. 从数据分析进一步延伸到产品功能设计和 A/B 实验评估方案。

---

## 项目局限性

1. 数据来源于 Fragrantica 平台，可能存在平台收录偏差；
2. 数据集中没有真实销量、价格、营销投放和渠道曝光信息；
3. 用户评分具有主观性，不同用户对香水偏好的差异较大；
4. 当前推荐系统基于气味文本相似度，不包含用户历史行为；
5. 新品潜力预测模型只能作为辅助参考，不能直接预测真实商业成功。

---

## 后续改进方向

1. 引入评论文本，分析用户对香水的具体评价情绪；
2. 加入价格、销量、品牌营销等外部数据，提高市场预测能力；
3. 构建更完整的用户行为推荐系统；
4. 使用 SQL 构建分析查询，提高项目对数据分析实习岗位的匹配度；
5. 将 Streamlit 应用部署到 Streamlit Community Cloud，生成可公开访问的在线 Demo；
6. 进一步增加 Dashboard 页面，展示核心指标和业务监控结果。

---

## 简历项目描述

**Fragrantica 香水市场分析与推荐系统项目**

基于 2.4 万条 Fragrantica 香水数据，使用 Python 完成数据清洗、探索性分析和可视化，构建 `Popularity Score` 综合热度指标，分析品牌、国家、Gender 与香调特征对用户热度的影响；基于 main accords 与前中后调文本特征，使用 TF-IDF 和余弦相似度实现相似香水推荐，并通过 Streamlit 封装为交互式网页应用；进一步构建新品香水市场潜力评估模块，基于上市前可知特征预测新品概念接近历史高热度香水的概率，并返回相似历史案例与可信度提示。

---

## Author

Yuanyuan Deng  
Undergraduate Student, Big Data Management and Application  
Interested in Data Analytics, Product Analytics, and Data-driven Applications
