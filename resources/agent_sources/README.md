# 实验助手本地原始资料与整理说明

本目录保存实验助手知识库所依据的原始资料。知识库不是直接把整份 PDF 发送给大模型，而是先人工整理为 `src/agent/knowledge_base/` 下的短主题条目，再进行本地检索，以减少 token 并提高回答可追溯性。

## 资料清单

### `GXNU_Michelson_Lab_Guide.pdf`

- 来源：广西师范大学物理实验中心，《迈克尔逊干涉仪的调整及使用》
- 在线地址：https://phylab.gxnu.edu.cn/_upload/article/files/4d/74/7131dc7b4941bc5439b9b628f603/f412637f-4112-4daf-ade8-dfe88ea44e55.pdf
- 页数：1
- 重点：实验目标、钠光波长测量、重复测量、空程差、慢速调节、光学元件保护。
- 对本项目的用途：基础操作、安全注意和波长测量流程。

### `TJNU_Michelson_Lab_Guide.pdf`

- 来源：天津师范大学，《迈克尔逊干涉仪调节与应用》
- 在线地址：https://www.tjnu.edu.cn/__local/A/30/45/49928BE253ECB0440DA7BD7CA70_AED7B962_394CC.pdf
- 页数：5
- 重点：分振幅双光束结构、等倾与等厚干涉、条纹变化规律、钠光双线和相干长度。
- 对本项目的用途：扩充实验原理、条纹类型和数据处理知识。

### `CMU_Lab7_Michelson.pdf`

- 来源：Carnegie Mellon University, `Lab 7: The Michelson Interferometer`
- 在线地址：https://www.cmu.edu/biolphys/smsl/teaching/IntermedOptics/IntOptics_data/lab%20manuals/Lab7%20-%20Michelson.pdf
- 页数：6
- 重点：白光条纹寻找、先用准单色光完成预调、微米头标定、激光与滤光片安全。
- 对本项目的用途：白光寻零步骤和安全诊断。该资料明确建议：先用准单色光把两臂调到接近白光干涉条件，再换白光并进行非常缓慢、细致的小幅调节。

### `NIST_Technical_Note_1297.pdf`

- 来源：NIST Technical Note 1297, `Guidelines for Evaluating and Expressing the Uncertainty of NIST Measurement Results`
- 在线地址：https://doi.org/10.6028/NIST.TN.1297
- 页数：25
- 重点：A 类和 B 类标准不确定度、合成标准不确定度、扩展不确定度、覆盖因子及结果表达。
- 对本项目的用途：约束智能体对误差与不确定度的解释，并作为后续确定性计算工具的依据。

## 面向本项目的综合结论

1. 白光寻零不能依赖高速大范围扫描。更可靠的流程是先完成光斑重合和准单色条纹预调，再换白光进行低速、小范围搜索。
2. 位移测量必须保持单方向读数或明确消除回程间隙；程序应记录运动方向和反向后的稳定区间。
3. 波长计算采用 `λ=2Δd/N`，但最终结果还必须考虑位移标定、读数分辨率、条纹识别、重复性和环境扰动。
4. 智能体只能基于当前状态和资料提出建议，不能把 YOLO 置信度直接解释成测量不确定度。
5. 涉及激光、镜片、机械调节和电机扫描的建议必须带安全边界；MVP 不允许直接控制电机。

## 更新规则

- 新资料必须记录机构、标题、在线地址、下载日期和用途；
- 优先使用大学、国家计量机构或仪器制造商资料；
- 原始 PDF 保留不改，整理内容写入知识库 Markdown；
- 每条知识应能追溯到至少一份本地原始资料；
- 不把整份 PDF 放入单次提示词。
