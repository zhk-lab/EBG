# 对齐式 Compact 的文献依据

本文只讨论**检索完成后，图信息如何组织成模型输入**，不讨论图在数据库中的存储格式。

## 核心原则

对齐式 Compact 包含五个原则：

1. 只输入与问题相关的小型局部图，不展开整张图。
2. 以直接命中的 Root 为中心，把相关邻居放在它附近。
3. 明确区分直接证据和结构上下文。
4. 用短关系行保留边的类型、方向和位置，并紧邻原始证据。
5. 最相关内容优先，同时删除无关 Behavior、内部 ID 和重复源码。

## 1. 只输入与问题相关的小型局部图

**论文依据。** G-Retriever 先根据问题选择相关且连通的小型子图，再把其中的节点和边 textualize 后输入模型。其实验表明，这种检索能大幅减少图文本 token；去掉 textualized graph 又会明显降低效果。这说明有效的压缩不是丢掉图，而是只保留与问题有关的子图。[G-Retriever](https://arxiv.org/html/2402.07630)

**BEG 对应。** 先固定少量直接命中的 Root，再选择与这些 Root 有关的一跳邻居。内容变短后直接节省 token，不能再用剩余预算补入无关 Root。

## 2. 以 Root 为中心集中组织邻居

**论文依据。** Talk Like a Graph 直接比较多种图文本编码，发现编码方式会显著影响推理结果。对于局部邻居问题，以目标节点为中心集中列出相邻节点的 incident 编码明显优于分散的边列表；论文将其归因于所需关系在文本中更集中、更容易访问。[Talk Like a Graph](https://arxiv.org/html/2310.04560)

GraphText 进一步把目标节点、不同跳数的邻居和属性组织成层次化图语法树。消融实验中，完整层次结构优于扁平序列和无序集合，说明“围绕目标组织上下文”本身有价值。[GraphText](https://arxiv.org/abs/2310.01089)

**BEG 对应。** 每个结果先展示 `[DIRECT ROOT]`，随后紧邻展示由该 Root 引出的 `[CONTEXT via calls/feeds]`，而不是按全局边顺序混排多个 Symbol。

## 3. 区分直接证据与补充上下文

**论文依据。** GraphText 的消融表明，保留节点、关系和属性之间的层次优于全部展平。LightRAG 的回答输入也分别组织实体、关系和原文块，说明成熟的 GraphRAG 系统通常会区分不同作用的信息；但 LightRAG 没有单独证明这种分区优于其他格式，因此它属于设计佐证，而不是因果实验。[GraphText](https://arxiv.org/abs/2310.01089)；[LightRAG](https://arxiv.org/html/2410.05779)

**BEG 对应。** `[DIRECT ROOT]` 表示由文档或搜索直接命中的源码，`[CONTEXT via ...]` 表示用于解释调用或数据流的邻居。模型无需猜测哪段是主要证据、哪段只是补充信息。

## 4. 保留明确关系，并让关系紧邻证据

**论文依据。** Talk Like a Graph 表明，节点和边在文本中的表达方式会改变模型能否正确读取图结构。G-Retriever 同时 textualize 选中节点和边，而且去掉图文本会损害效果。这共同支持：不能只提供若干互不关联的内容块，仍需保留简短、明确的关系。[Talk Like a Graph](https://arxiv.org/html/2310.04560)；[G-Retriever](https://arxiv.org/html/2402.07630)

**BEG 对应。** 使用一行表达方向、类型和位置，并在其后立即放置对应源码：

```text
[DIRECT ROOT]
shortuuid/django_fields.py::ShortUUIDField.__init__@14-25
<Root 原始源码>

[CONTEXT via calls]
ShortUUIDField.__init__@23 calls ShortUUIDField._generate_uuid@27-31
<邻居中对应 Behavior 的原始源码>
```

源码和行号负责提供可核验的事实，关系行负责解释两段源码为什么同时出现；内部 Behavior/Evidence ID 不需要进入模型输入。

## 5. 最相关内容优先，并删除重复信息

**论文依据。** GraphRAG 在回答阶段按 helpfulness 排序社区报告，再交给模型处理；这说明相关性顺序是模型输入结构的一部分。[GraphRAG](https://arxiv.org/html/2404.16130) 长上下文研究也发现，关键信息位于开头或结尾时通常比埋在中间更容易被利用。[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

**BEG 对应。** 直接 Root 排在结构邻居之前；邻居只保留边能对应到的 Behavior，无法可靠判断时才回退到完整 Symbol；同一源码行只展示一次。




> 对齐式 Compact 把图转换成以 Root 为中心、按证据作用分区、关系显式且无重复的紧凑文本。它删除的是 JSON 嵌套、内部 ID、重复源码和低相关分支，而不是任务所需的源码与图关系。

这些论文为其设计原则提供依据；BEG 自身的消融实验则负责验证这种组合在代码定位任务上是否比 JSON 展开使用更少 token，并保持任务效果。

