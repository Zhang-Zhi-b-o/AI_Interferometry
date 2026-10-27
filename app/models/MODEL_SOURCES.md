# 模型版本与来源

Qt 工作台（`app/main.py`）默认使用下列全量模型，可在实时页面切换人工 600 张模型、
历史模型或自定义权重。`models/current/best.pt` 保留历史权重，未覆盖。

## Qt 工作台模型（2026-10-27）

两者均为 YOLOv8n，类别 `0=zero_order`、`1=near_fringe`，训练尺寸 960。

| 页面名称 | 发布文件 | 人工测试集 mAP50-95 | SHA-256 |
| --- | --- | --- | --- |
| 全量 2954 张（默认） | `models/releases/yolov8n_full2954_20261027.pt` | 0.84617 | `40345c9fe113b35845432872aefd35ee196438258fd0edb401c3b8f9a338a677` |
| 人工确认 600 张 | `models/releases/yolov8n_human600_20261027.pt` | 0.81776 | `7ba20b14c18833ccc727d724a1107c3c0a18e5e3b1e1e4d4723479fd58fa179d` |

来源均在 `1026新数据集/yolov8_dataset_20261026/`：全量模型来自
`training_full2954_20261027/runs/yolov8n_full/weights/best.pt`，由官方 COCO
YOLOv8n 初始化；600 张模型来自 `training600_20261027/runs/yolov8n_human600/weights/best.pt`。
全量数据包含 600 张人工确认与 2354 张自动标签；实际训练 1927 张、人工验证 100 张、
人工测试 100 张，另 827 张因时间隔离留作储备。600 张模型使用相同人工验证/测试集。
验证集 mAP50-95 分别为 0.86236 和 0.80708，推荐全量模型。测试集来自同一批视频，
尚非独立采集场景的泛化测试。完整对比见上述全量训练目录的 `comparison.csv`、`comparison.json`。
程序推理尺寸及阈值仍可在推理参数中调整，表中数值是统一尺寸 960 的离线评估结果。

## 当前模型（2026-08-11）

- 文件：`models/current/best.pt`
- 版本名：`yolov8n_all_human_confirmed`
- 来源：`D:\Files\Work\光电\AI_Interferometry_prepare\make_dataset_new\combined_human_model_training\runs\yolov8n_all_human_confirmed\weights\best.pt`
- 数据：合并后的全部人工确认双类别数据，`0=zero_order`、`1=near_fringe`
- 训练：YOLOv8n，`imgsz=640`、`batch=16`、最多 120 轮、`patience=18`
- 最佳轮次：110
- 验证：Precision 0.89561、Recall 0.91043、mAP50 0.95146、mAP50-95 0.77360
- SHA-256：`00F49460D29980B13BD8D9CE84233658118F95EFAFAC745A6F97E05984C7E31C`

## 历史模型（2026-08-10）

- 文件：`models/history/pre_annotation_yolov8n_pool_after_1240_best.pt`
- 版本名：`pre_annotation_yolov8n_pool_after_1240_best`
- 来源：`D:\Files\Work\光电\AI_Interferometry_prepare\make_dataset_new\pre_annotation_model_training\pre_annotation_yolov8n_pool_after_1240_best.pt`
- 数据：`dataset/pre_annotation` 训练池，约 646 张图，其中 50 张背景图
- 训练：YOLOv8n，`imgsz=640`、`batch=8`，最佳轮次 28
- 验证：Precision 0.850、Recall 0.748、mAP50 0.895、mAP50-95 0.660
- SHA-256：`526D9F464F80C30981FF8EDE1DCFC27A53490AE371FDE2E719F25088E49D6E45`

## 早期候选模型

以下权重于 2026-07-20 随提交 `218f802` 纳入仓库。仓库历史没有记录它们的完整
训练命令或数据绝对路径，因此只保留可核验的信息，不推测训练来源。

| 文件 | 可确认来源 | SHA-256 |
| --- | --- | --- |
| `models/candidates/mixed_real_20250628.pt` | 文件名标识为 mixed real 候选；曾作为初始 `models/current/best.pt` | `BDCFA8BA138D9F2DAB706395DD6E6789AD2C0F8B1ECF01CAC42F9104CA454130` |
| `models/candidates/yolo11s_realistic_20250628.pt` | 文件名标识为 YOLO11s realistic 候选 | `35B18926ED700141D512658577D7981E611E2C03640DF399212FF03F99571C30` |
| `models/candidates/yolov8n_sim_20250628.pt` | 文件名标识为 YOLOv8n simulation 候选 | `0BA74A53C4319ACF0919D07C6482ED7794F83B22D2CAD0FA546D47090ECD0D95` |
| `models/yolov8_interference.pt` | 早期干涉条纹权重；详细训练来源未记录 | `C4A732DE884ECE316A2ECA0C6B88D405A1A6CA58F04C26D1FB6539EEFDD8FE6B` |

未记录完整来源的早期模型只用于回溯和对照，不应直接替换当前模型。
