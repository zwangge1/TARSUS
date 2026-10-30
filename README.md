# TARSUS

Taxon-Aware Ratio priors for Skeletal pose Under featherS：在 DeepLabCut 3 上加入按物种条件化的鸟腿骨长比先验，自标注视频里看不见的膝和髋。

- 方案：`docs/design/06_DLC改造方案_2026-10-30.md`
- 预注册：`docs/prereg/`（判据先于结果写入，带日期）
- 官方 DLC：`third_party/DeepLabCut/`（来源与修改规则见 `third_party/UPSTREAM.md`）

## 目录
| 路径 | 内容 |
|---|---|
| `third_party/DeepLabCut/` | 官方 DLC，原样导入后按 M1 到 M7 逐条修改 |
| `prior/` | 构建物种先验 `prior_v0.json`（产品名 tarsus-prior） |
| `configs/` | 实验组 A 到 D 的配置 |
| `scripts/` | 建数据集、训练、评测 |
| `outputs/` | 产物，不进 git |

## 数据（只引用，不复制）
- Duong DLC 项目与原始视频：`FFY/duong_data/`
- 骨长：`EvoWM/data/raw/{watanabe2017,eliason2023,skelevision}/`

## 安装（服务器）
`pip install -e third_party/DeepLabCut`

## 同步
本地 `.\sync.ps1 "说明"` → GitHub → A100 `git pull`。不在挂载盘上运行 git。
