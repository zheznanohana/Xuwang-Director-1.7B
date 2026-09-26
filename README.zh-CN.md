# 虚妄决策模型 · Xuwang-Director-1.7B

[Hugging Face 模型](https://huggingface.co/nanohana233/Xuwang-Director-1.7B) · [训练数据](https://huggingface.co/datasets/nanohana233/Xuwang-Director-Data) · [完整 GitHub Release](https://github.com/zheznanohana/Xuwang-Director-1.7B/releases/tag/v0.1.0)

**不是让 AI 替我们做游戏，而是为游戏训练一个导演。**

虚妄是一个基于 Qwen3-1.7B-Base 微调、蒸馏的本地游戏决策模型。它不通过生成聊天文本来表达决定，而是用 193 个类型化输出头，直接为十类游戏任务输出候选概率。模型给出偏好，游戏规则负责约束、校验与执行。

## 它能做什么

- 调整下一场战斗的难度与属性。
- 配置遭遇敌人、奖励卡牌和最终宿敌。
- 选择 Boss 战术。
- 选择音乐场景编配参数与乐谱字段。
- 评估上一场战斗，并选择遭遇呈现的风格标签。

这是本作专用模型，不是通用聊天机器人，也不是所有游戏都能直接接入的通用导演。剧情对白生成不在这十类决策中。

## 为什么做本地版本

推理时不需要第三方云端 API 或 API Key；GGUF 权重和决策头随应用部署，模型版本由开发者固定。模型不是唯一裁判：预算、兼容性和跨字段约束依然由游戏处理。

## 下载与使用

到 [Releases](https://github.com/zheznanohana/Xuwang-Director-1.7B/releases) 下载：

1. `Xuwang-Director-1.7B-Q4_K_M.gguf`
2. `heads.json`

两者缺一不可。1.7B 表示底座参数规模，不是模型文件大小。Q4 文件大小为 1,107,408,416 字节。

安装 Python 3.10+、本仓库 requirements.txt 中的依赖，以及兼容的 llama.cpp 后，启动本地服务：

```sh
llama-server -m models/Xuwang-Director-1.7B-Q4_K_M.gguf --embeddings --pooling last --host 127.0.0.1 --port 18739 -c 1024 -b 1024 -ub 1024
```

另开终端运行：

```sh
python infer.py --heads models/heads.json --example examples/camp.json
```

示例返回营地评估的四组概率和选择。另附遭遇风格示例 `examples/flavor.json`。最小示例只计算决策头输出，不运行完整游戏的采购搜索与联合校验；其他通道需要严格遵循训练时的输入格式。

## 公开材料

- Q4 GGUF、193 个决策头、step-2000 LoRA 与 tokenizer。
- step 2000 / 2800 / 3000 三份研究检查点。
- 34,944 条合成教师数据（训练 25,951 / 验证 4,512 / 测试 4,481）。
- 离线评测、量化报告、训练代码与最小推理示例。
- SHA-256 校验值。

不包含完整游戏、美术音乐资产、账号配置、密钥及私人日志。

## 数据怎样解读

- **0.7719**：测试集上与教师集成的宏平均一致性；不是玩家满意度或战斗胜率。
- **0.9842**：量化版与全精度参考模型的字段一致性；不是任务正确率。
- 冻结底座对照为 0.7273，四种特征 MLP 为 0.7648–0.7657。
- 游戏中的原样合法率、策略搜索后的接受率，属于不同处理阶段；完整系统效果包含游戏约束层的贡献。

2026-09-27 发布验证：9 项单元测试通过；待发布 GGUF 的两类示例与命令行入口实跑通过；392 个 LoRA 张量及全部决策头与选定 step 2000 完全匹配。详见 LOCAL_VALIDATION.md。

## 来源与许可

本仓库原创代码和文档采用 Apache-2.0。底座来自 Qwen3-1.7B-Base，保留其 Apache-2.0 许可。训练标签来自游戏规则、提供商决策模型、Qwen 聊天教师及 Boss 搜索教师，数据保留教师来源字段。代码许可的范围与材料来源见 MODEL_TERMS.md、DATA_CARD.md。
