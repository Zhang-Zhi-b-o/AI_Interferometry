# 模型版本与来源

运行时始终读取 `models/current/best.pt`。替换模型时保留原始文件名、来源目录和
SHA-256，避免仅凭 `best.pt` 无法判断训练批次。

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
