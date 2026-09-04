# 如何让 LLM 更好地读取行为证据图

## 1. 结论

LLM 最难的并不是理解一条边，而是从线性 token 中恢复整张图，并在多条分支之间持续记住“走过哪里、为什么走、还缺什么”。因此，增强读图能力不能只靠把 JSON 字段解释得更详细，也不能把更多跳邻居一次性塞进 Prompt。

现有研究最支持以下方案：

```text
完整 BEG 保留在后台
→ 根据当前问题检索小而连通的局部图
→ 程序完成邻接查询、跳边、去环和路径闭合
→ 模型一次读取少量有序路径
→ 每个 Behavior、关系边和支撑原文紧邻出现
```

模型仍然是在“读图”：节点、边、方向和路径都被显式保留；但它读的不是完整图或分散的 `nodes/edges/evidence` 表，而是**查询相关、按路径排列、同时带原始证据的局部图**。

对 BEG 最匹配的不是某一篇论文的完整实现，而是三类研究的组合：

- **RepoGraph** 说明代码图适合导航和定位，也直接证明“更多跳数、更多图信息”可能降低效果；
- **GRAG** 说明拓扑视图和原文视图应同时提供，不能让图摘要替代原始语义；
- **GraphReader、StructGPT、ToG/RoG** 说明跨边遍历应主要由程序执行，模型负责选择方向、判断证据是否充分和生成答案。

## 2. 为什么直接读图会损失语义

### 2.1 图最终仍要线性化

LLM 接收的是 token 序列，不是真正的邻接结构。同一张图使用不同编码和排列，效果可能相差很大。[Talk Like a Graph](https://proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf) 发现图编码方式可带来 4.8 至 61.8 个百分点的差异；按当前节点聚合邻边通常比全局边列表更容易理解。

这意味着当前这种结构并不中立：

```json
{
  "behaviors": [],
  "source_contexts": [],
  "edges": []
}
```

模型必须先在 `behaviors` 中理解节点，再去 `edges` 查端点，最后回到 `source_contexts` 找原文。一次判断需要反复跨区域按 ID 做 join，正是容易丢失语义的地方。

### 2.2 图越大不一定越好

[G-Retriever](https://proceedings.neurips.cc/paper_files/paper/2024/file/efaf1c9726648c8ba363a5c927440529-Paper-Conference.pdf) 在 WebQSP 上将平均约 1371 个节点、10 万 token 的原图检索为约 18 个节点、610 token 的连通子图。其一个实验中，直接输入完整 textual graph 的结果甚至低于只给问题；检索后的连通子图明显更好。该方法包含训练组件，数值不能直接外推到 BEG，但它清楚说明：**信息更多不等于模型利用得更好，相关且连通才重要。**

与 BEG 最接近的 [RepoGraph](https://arxiv.org/abs/2410.14684) 也得到相同结论：2-hop 子图虽然包含更多信息，直接摊平后的任务完成率却只有 26%，是所有变体中最差，甚至低于原 baseline。论文还发现图对文件级定位提升更明显，而最常见错误仍是上下文语义没有对齐。这与 BEG 目前“定位较好、语义较弱”的现象高度一致。

### 2.3 多跳和分支会放大错误

[Can LLMs Perform Structured Graph Reasoning?](https://arxiv.org/abs/2402.01805) 发现，节点可选分支越多，模型的图遍历表现越差；其 PathCompare 方法优于普通 Chain-of-Thought。[NLGraph](https://proceedings.neurips.cc/paper_files/paper/2023/file/622afc4edf2824a1b6aaf5afe153fa93-Paper-Conference.pdf) 也显示，路径变长、图任务变复杂后，CoT、few-shot 等通用提示方法不能稳定解决问题。

因此，不能把下面的工作交给不开思考模式的模型：

```text
选一条边 → 找目标节点 → 回查证据 → 再选边
→ 记住此前路径 → 比较另一条分支 → 判断何时停止
```

只在 Prompt 中写“请沿边逐步检查”并不能根治问题。

### 2.4 顺序本身影响理解

[Graph Description Order](https://aclanthology.org/2025.acl-long.321/) 系统比较了随机、BFS、DFS、PageRank 和 PPR 等排列，发现有序描述普遍优于随机顺序，而且不同任务需要不同顺序。局部连接和可达性更适合从 root 向外的 BFS；已经确定的深层因果或调用链更适合将整条路径连续展示。

所以局部图不能按全局 ID、构图时间或 JSON 容器顺序排列。应先显示 root，再连续显示与问题有关的完整路径。

## 3. 哪些研究最适合 BEG

### 3.1 RepoGraph：最贴近 Repo 场景

RepoGraph 从查询词提取 k-hop 代码 ego-graph，并将其作为 Repo 导航。它提高了文件定位，也减少了 Agent 平均轮数；但 2-hop 平铺图表现最差。对 BEG 的直接启示是：

1. 图首先适合做精确导航，而不是让模型阅读所有关系；
2. 一跳应是默认上限，更远关系只有在补齐当前行为路径时才加入；
3. 图必须和原有语义上下文一起提供，不能只提供依赖关系；
4. “找到 Gold 文件”不等于“理解 Gold 行为”，二者必须分别评测。

### 3.2 GRAG：最适合解决语义损失

[GRAG](https://aclanthology.org/2025.findings-naacl.232/) 先检索与问题相关的 textual subgraph，再向模型提供两个互补视图：

- **graph view**：节点、边、方向和连接路径；
- **text view**：节点和边对应的原始文本语义。

这正好对应 BEG 的需要：Behavior 与 `calls/feeds/precedes/continues` 保留图关系，Code/Trace Evidence 提供最终语义和定位。二者必须在同一次 `read` 中对齐，不能只给图，也不能退化成只有连续源码的文件包。

### 3.3 GraphReader 与 StructGPT：最适合交互方式

[GraphReader](https://arxiv.org/abs/2406.14550) 不把长图一次性交给模型，而是让 Agent 用固定接口从粗到细读取节点内容和邻居，并持续记录已获得的信息。[StructGPT](https://aclanthology.org/2023.emnlp-main.574/) 同样把结构化数据留在外部系统中，由接口取回局部关系，再让模型推理。

BEG 可以保留简单的 `query/read/finish`，但 `read` 后的邻接查询、路径状态、去环和证据闭合应由程序完成。Luna 只需要判断当前局部图是否已经回答任务，以及是否还需读取一个 frontier。

### 3.4 ToG 与 RoG：路径优于无序邻居

[Think-on-Graph](https://proceedings.iclr.cc/paper_files/paper/2024/file/10a6bdcabbd5a3d36b760daa295f63c1-Paper-Conference.pdf) 由程序维护候选路径和 beam，模型只筛关系、实体并判断是否停止。[Reasoning on Graphs](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e2aeb66481dd63a32421bf032b70384-Paper-Conference.pdf) 则先规划关系路径，再由图执行器检索真实可达路径。

BEG 不需要照搬它们的训练或多次 LLM 剪枝，但应借用其关键原则：**程序返回已经验证的完整候选路径，而不是让模型从一堆邻居中自行拼路。**

## 4. 建议的 BEG 读图方式

### 4.1 后台检索

`read(F)` 或精确 `query` 先确定 root 文件和代码作用域。检索器随后：

1. 取出 root 作用域内的完整 Behavior；
2. 检查 Behavior 的 trigger、operation、result 是否闭合；
3. 只有当 `calls/feeds` 直接支撑该 Behavior，或目标作用域负责必要结果时，才自动补入邻居；
4. 程序维护方向、visited、去环和真实边校验，不让模型手工追踪 Evidence ID；
5. 默认只展开一跳；更远节点只有在一跳仍无法形成完整路径时才按需加入；
6. 以“少量完整路径”而不是“最多节点数”为主要预算单位；预算外关系进入 frontier。

对于 Trace，同样由程序把反馈、当时行为和同一对象的后续行为连接成连续路径；`precedes` 只表示时间顺序，`continues` 只表示同一对象延续，不能自动改写成因果。

### 4.2 候选 A：递归 JSON

旧示例中的 `root_scope → hops → edge → target_scope` 包装过多，而且同时复制 Evidence `content` 和完整 `source`。递归 JSON 候选将其收敛为：每个代码作用域只有 `location`、`behaviors`、`source` 和 `next`；`next` 中的关系直接连接下一个作用域。它只是待比较的程序友好基线，不是已经验证的最终格式；另外两种候选见 `graph_input_formats.md`。

```json
{
  "root": {
    "location": {
      "path": "src/api.py",
      "symbol": "Client.run",
      "lines": [18, 40]
    },
    "behaviors": [
      {
        "behavior_id": "B0107",
        "result_type": "return",
        "trigger": [{"evidence_id": "E12", "lines": [20, 20]}],
        "operation": [{"evidence_id": "E13", "lines": [24, 24]}],
        "result": [{"evidence_id": "E14", "lines": [31, 31]}]
      }
    ],
    "source": "18 | <连续原始源码>\n...\n40 | <连续原始源码>",
    "next": [
      {
        "relation": "calls",
        "evidence": [{"evidence_id": "E13", "lines": [24, 24]}],
        "target": {
          "location": {
            "path": "src/output.py",
            "symbol": "emit",
            "lines": [8, 19]
          },
          "behaviors": [],
          "source": "8 | <连续原始源码>\n...\n19 | <连续原始源码>",
          "next": []
        }
      }
    ]
  },
  "frontier": [
    {
      "from": "src/output.py::emit",
      "relation": "feeds",
      "to": "src/cache.py::store",
      "read_id": "F0003"
    }
  ]
}
```

该结构只有两种递归对象：

- **代码作用域**：`location + behaviors + source + next`；
- **关系**：`relation + evidence + target`。

`path + symbol` 只是代码中确定存在的定位字段，不新增 Symbol 节点或 Symbol ID。第一层是 root；继续沿 `next.target` 读取就是下一跳。默认局部图只有一跳，因此通常只有一层 `target`；只有程序确认需要补齐当前行为路径时才继续嵌套。

关系项必须包含 `target` 或 `target_ref` 二者之一：首次到达某个作用域时使用完整 `target`；该作用域已经在本次局部图中出现时，仅使用 `target_ref: "path::symbol"`，防止重复发送同一源码。

### 4.3 原文放置与聚合规则

后台图和模型输入承担不同职责：

- **后台 `behavior_graph.json`** 保持规范化：每条 Evidence 的逐字 `content` 只在全局 Evidence 中保存一次；每个连续源码作用域也只保存一次；Behavior 和 Edge 只引用 Evidence ID。
- **模型读取的 Local Graph** 不再重复 Evidence `content`。Repo 原文只在所属 `(path, symbol, lines)` 的 `source` 中出现一次；Behavior 的 trigger、operation、result 以及 Edge 的 evidence 只给出 ID 和精确行号，且与该 `source` 放在同一个作用域对象内。

Repo 的确定性聚合顺序为：

1. 按完全相同的 `(path, symbol, symbol_lines)` 聚合代码；
2. 该作用域的连续带行号源码只写入一个 `source`；
3. 该作用域的全部 Behavior 按结果行排序，分别标出 trigger、operation 和 result 的 Evidence 行号；
4. 从该作用域出发的同类、同目标边合并，支撑 Evidence 去重；
5. `next` 按“与当前 Behavior 直接相交、`feeds`、`calls`、目标位置”排序；
6. 程序维护 visited，禁止成环。同一目标再次出现时只给可读的 `target_ref`，不重复源码；
7. 没有展开的邻居只进入 `frontier`，不混入当前路径。

Trace 没有连续源码，因此原始事件只在所属交互的 `events[].content` 中按 turn 出现一次；Trace Behavior 和 `precedes/continues` 边只引用 Evidence ID，不复制事件原文。

这样，“原文紧邻 Behavior 和 Edge”指的是它们处于同一个作用域块中，并通过精确行号直接对齐，不是把同一段原文复制三遍。

### 4.4 Prompt 只解释一种阅读顺序

Prompt 不应要求模型“自由遍历图”，而应明确：

1. 先读 `root.behaviors`，再在紧邻的 `root.source` 中核对相应行；
2. 按 `next` 顺序读取关系和 `target`，不自行搜索或猜测其他边；
3. trigger、operation、result 共同组成 Behavior；节点和边都是中立事实，不表示一定存在 SpecGap 或 SilentSwap；
4. Repo 的 `source` 和 Trace 的 `events[].content` 是最终语义与定位依据；
5. 当前路径无法回答问题时才读取 `frontier`。

## 5. 不建议采用的做法

- 不一次输入完整 `behavior_graph.json`；
- 不默认展开完整二跳图；
- 不分别列出全局节点表、边表和 Evidence 库；
- 不把一组互不连接的 top-k Behavior 当作子图；
- 不让模型逐个选择每一跳并自己维护 visited；
- 不用 LLM 生成的行为总结替代 Code/Trace 原文；
- 不因长上下文窗口更大就增加无关节点。

## 6. 最小验证实验

在同一 AgentLoop、样本、模型和 Judge 下比较：

1. Raw baseline；
2. 当前分离数组 Local Graph；
3. 路径化 Local Graph，但不含连续原文；
4. 路径化 Local Graph + 紧邻原始 Evidence/source；
5. 完整二跳平铺图。

必须同时记录：语义分、定位分、总 token、动作数、实际读取 Gold 的轮次、路径中的无关节点比例和失败类型。再做四个针对性消融：

- root-first 顺序与随机顺序；
- 默认一跳与平铺二跳；
- 程序自动闭合路径与模型手工逐边读取；
- graph-only 与 graph + raw evidence。

如果第 4 项不能稳定高于第 2 项，说明问题不只是图的序列化，而可能是 Behavior 抽取或边覆盖本身不完整；不能继续通过增加字段掩盖底层问题。

## 7. 最终建议

BEG 下一版应采用：

```text
排序目录确定 root
→ 程序检索小而连通的局部图
→ 程序将其整理成少量、方向明确的完整 Behavior 路径
→ 每个节点旁同时给图语义和原始 Evidence/source
→ 模型判断，必要时再读取 frontier
```

这不是把图重新压成普通文本，而是把完整图变成**可由序列模型稳定阅读的路径化局部图**。它保留 BEG 已经表现出的定位优势，同时减少模型手工跳边和跨表拼接，从机制上最有希望解决当前语义分下降的问题。

## 参考研究

- [RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph](https://arxiv.org/abs/2410.14684)
- [GRAG: Graph Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-naacl.232/)
- [GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of LLMs](https://arxiv.org/abs/2406.14550)
- [StructGPT: A General Framework for Large Language Model to Reason over Structured Data](https://aclanthology.org/2023.emnlp-main.574/)
- [Talk Like a Graph: Encoding Graphs for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf)
- [Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](https://aclanthology.org/2025.acl-long.321/)
- [G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering](https://proceedings.neurips.cc/paper_files/paper/2024/file/efaf1c9726648c8ba363a5c927440529-Paper-Conference.pdf)
- [Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph](https://proceedings.iclr.cc/paper_files/paper/2024/file/10a6bdcabbd5a3d36b760daa295f63c1-Paper-Conference.pdf)
- [Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e2aeb66481dd63a32421bf032b70384-Paper-Conference.pdf)
- [Can LLMs Perform Structured Graph Reasoning?](https://arxiv.org/abs/2402.01805)
- [NLGraph: Benchmarking Graph Reasoning with Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/622afc4edf2824a1b6aaf5afe153fa93-Paper-Conference.pdf)
