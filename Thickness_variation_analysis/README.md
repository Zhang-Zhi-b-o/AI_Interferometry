# 单张彩色干涉图厚度分布分析

该工具从一张白光彩色干涉图生成薄膜区域掩膜、相对厚度云图、厚度叠加图、截面曲线、置信度图、CSV 数据和中英文兼容的 HTML/Markdown 报告。

## 科学边界

没有无薄膜基准和“颜色—光程差”标定时，单张 RGB 图不能唯一确定绝对厚度。默认结果是**模型依赖的相对厚度估计**：程序将连续彩色条纹展开为条纹级次，并假设一个完整颜色周期对应一个有效干涉级次。它适合观察厚薄起伏、均匀度和异常区域，不应直接作为可溯源的精密厚度结果。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 快速运行

```powershell
python analyze.py example_input.jpg -o example_output --refractive-index 1.523 --wavelength-nm 589.3
```

若已知样品平均厚度为 176.7 μm：

```powershell
python analyze.py example_input.jpg -o example_absolute --reference-thickness-um 176.7
```

此参数只设置整张相对图的厚度零点，不会提高相对起伏的测量准确度。

如果自动分割不理想，可手工指定 ROI：

```powershell
python analyze.py image.jpg -o output --roi 100,60,500,420
```

如果厚薄方向与已知事实相反，加 `--invert`。

## 定量颜色标定模式

标定 CSV 需要包含：

```csv
opd_um,r,g,b
-2.0,80,120,210
-1.5,95,180,170
-1.0,180,160,90
```

其中 `opd_um` 必须是已经扣除无膜基准的薄膜附加实际光程差，而不是动镜位移。运行：

```powershell
python analyze.py image.jpg -o calibrated_output --calibration colour_opd.csv --refractive-index 1.523
```

程序采用

\[
t=\frac{\Delta OPD}{2(n-1)}
\]

换算厚度。制作标定表时，只使用电机停车、条纹清晰后的图像，并固定曝光、白平衡、增益和光源亮度。

## 输出

- `thickness_map.png`：厚度云图；
- `thickness_overlay.png`：厚度伪彩叠加；
- `profiles.png`：横纵截面曲线；
- `confidence_map.png`：分析置信度；
- `sample_mask.png`：有效样品区域；
- `thickness_map_um.csv`：逐像素厚度数据；
- `summary.json`：统计结果；
- `report.md`、`report.html`：自动分析报告。

## 推荐的正式实验流程

1. 固定曝光、白平衡、增益和光源；
2. 电机逐步移动，每次停车后读取动镜位移并拍清晰图；
3. 用已有模型把动镜位移换算为实际光程差；
4. 建立同一相机条件下的 `opd_um,r,g,b` 标定表；
5. 拍摄无薄膜基准图并扣除系统背景；
6. 再分析不均匀薄膜图。

