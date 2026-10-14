# RoboDojo → Jev / Valen 标注样例

这是一批 **模型检查画面后制作、待人工复核** 的离线标注初稿。12 个样例来自 4 条纯 π₀.₅ 仿真轨迹：6 个请求检查、5 个正常进行或子目标完成、1 个证据不足。不是已验证的训练真值，也没有运行 Jev 准确率实验。

打开 [review.html](review.html) 查看片段、时间点图集、每个视角的原始输入图片及标注理由；[samples.jsonl](samples.jsonl) 是可供 Valen 读取的格式。

## 样例

所有时间都是从 episode 开始计算的仿真秒。标签描述窗口末尾应作出的判断，不是整段视频的最终成败。

| ID | 任务 | 观测时间 | 初稿标签 | 要检验的区别 |
| --- | --- | --- | --- | --- |
| p01 | organize_table | 0–2 | 继续 | 抓取正在进行，尚未完成不等于失败 |
| p02 | organize_table | 1–5 | 子目标完成 | 闹钟正常释放后有稳定支撑 |
| p03 | organize_table | 6–9.2 | 请求检查 | 摆件释放后倾倒在桌面 |
| p04 | put_bottles_into_dustbin | 14–17 | 继续 | 瓶子抬起并开始换手 |
| p05 | put_bottles_into_dustbin | 16–19 | 请求检查 | 换手后瓶子脱离两只夹爪 |
| p06 | put_bottles_into_dustbin | 22–28 | 请求检查 | 反复夹取但瓶子没有离开桌面 |
| p07 | classify_objects | 8–9.6 | 继续 | 玩偶正在被搬运 |
| p08 | classify_objects | 12–14 | 请求检查 | 抓取没有维持，玩偶留在桌面 |
| p09 | classify_objects | 14–18 | 子目标完成 | 重新抓取成功，不重复报告旧失误 |
| p10 | classify_objects_by_language | 8–10 | 请求检查 | 绿车仍在桌面，篮内黄车不能替代目标 |
| p11 | classify_objects_by_language | 8–16 | 请求检查 | 多轮尝试仍未把绿车送入篮子 |
| p12 | put_bottles_into_dustbin | 18，单帧左腕 | 证据不足 | 近景遮挡且没有历史，不能判定换手结果 |

## 数据与标注约定

- 数据格式依照 [Valen/docs/data-format.md](https://github.com/Liuziyu77/Valen/blob/main/docs/data-format.md)。读取时的文档及 schema Git blob SHA 记录在 `sources.json`。
- 每条记录含 `group_id`、`request.state.messages`、`request.questions`、`targets`、`assets` 和 `meta`。图片路径相对 JSONL 所在目录，manifest 包含 SHA-256。
- `execution_state` 为五选一 Choice；`request_review`、`subgoal_complete` 为 Noul，标签使用完整的 `true` / `false` 概率分布，不使用推理返回值字段 `noul`。
- p12 仅监督 `insufficient_evidence`，省略两项二元标签。没有把“不知道”标成 false，也没有伪造 0.5 概率。它是明确记录的单视角、单帧消融样例，不代表默认三视角输入不足。
- 当前子目标、目标指代和成功条件均由离线标注者补充，用于检验“给定正确上下文”的能力。纯 π₀.₅ 日志只有原始任务，不声称这些细粒度动作曾输入策略，也不声称原策略能接受这些指令。`context_is_oracle=true` 标识此限制。
- 输入只包含截至决策时刻的指定观测。中文理由、异常边界、终局成功信息不进入模型 prompt。原生结果仅在来源清单中记录。
- `request_review=true` 表示值得低成本重新观察或规划检查，不等于急停、不可恢复失败、或已证明介入有收益。这里未执行真实干预，也没有反事实收益标签。
- 采用无历史报警/冷却的局部决策约定；片段末尾已恢复则不因窗口开头的旧失败重复触发。
- 正常释放与失误释放按任务目的地及物体支撑关系区分。可见持有失败不进一步推断打滑、主动松夹、接触力或故障原因。

## 时间、媒体与可追溯性

`sensors.mp4` 有初始帧，每个实际控制步增加一帧；本批核验 25 fps、控制周期 0.04 s、视频帧数等于最终步数加一。因此 `frame_index = step`，`sim_time_s = step × 0.04`。

模型输入从原始 `observations/NNNNNN.npz` 提取三相机 RGB，保持 640×480，无损 PNG。它们没有标注文字或标签覆盖。每条记录给出观测路径、源 NPZ SHA-256、步号和相机名。主视角缩略图仅供复核；连续视频由已有三相机录像按精确帧区间裁切，末帧包含在内。录像继承源记录器每视角 640×360 的显示缩放，不作为本批模型输入。

异常边界由已检查的离散观测给出，不声称是逐帧精确物理起点。`first_confirmed_s` 是所检查帧中首次确认或保守选定的复查时间；p06/p11 的重复尝试时刻尤其需要人工确认。`review_window_s` 是初稿建议窗口，不是验证过的最晚有效干预时刻。换手失误在 18.6 s 可确认，但当前 25 步一次的在线监测最早到 19 s 才有机会看到。

本批使用事件附近的非均匀采样和最多 8 秒历史，并不完全复现线上固定端点/历史长度。要报告在线召回和延迟，应重新按部署采样率回放完整 episode，保留连续正常时段，而非直接用这些精选样例估计线上误报率。

## 划分与验证

同一 episode 的正常、失败、恢复、重叠片段及消融视角共用一个 `group_id`。这 4 组只用于样例审阅，未划分 train/validation/test；以后同一 episode 及相同场景来源的变体应整组划分，不能随机拆相邻片段。

`inventory.json` 记录所选纯 π₀.₅ 批次的 10 条轨迹。第一轮检查了其中 5 条的稀疏主视角概览，并对入选 4 条的候选片段检查更密时间点及三视角；未声称逐帧看完所有 rollout。

重建（需要 NumPy、Pillow、ffmpeg、ffprobe）：

```bash
python datasets/jev_robodojo_pilot/build.py --source-root /path/to/robodojo-ten-pi05-only-20261013
```

在 Valen 环境内执行官方结构检查：

```python
from valen.data.schema import read_jsonl
records = read_jsonl('/absolute/path/to/datasets/jev_robodojo_pilot/samples.jsonl')
print(len(records))
```

结构通过不代表标签已获人工认可，也不代表已通过模型 processor 的视觉 token/长度限制。此批未训练、未调用模型推理、未上传数据。
