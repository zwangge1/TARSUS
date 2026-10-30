# third_party/DeepLabCut 来源

- 仓库：https://github.com/DeepLabCut/DeepLabCut
- 分支：main
- commit：fa5a8c36100c85300b7d4a534deb38d7bd8b2fa8（2026-09-29，"Docs: add missing TOC entry for napari docs (#3533)"）
- 许可：LGPL-3.0（见 DeepLabCut/LICENSE）。修改过的文件保留原版权头，修改记录见 git 历史。
- 收录范围：`deeplabcut/`、`tests/`、`conda-environments/`、`pyproject.toml`、`setup.py`、`LICENSE`、`AUTHORS`、`NOTICE.yml`、`README.md`。未收录：`examples/`（74 MB 示例数据）、`docs/`、`dev-docs/`、`docker/`、`tools/`、`.github/` 等，与代码运行无关。
- 规则：本仓库第一个提交是上述文件的原样拷贝。之后对 DLC 的每一处修改（方案 06 的 M1 到 M7）单独提交，提交信息以 `[DLC-Mx]` 开头，`git diff <首个提交> -- third_party/` 即可逐行对照官方。
