# 学習用 Brush 0.3.0

公式タグ v0.3.0 のコミット `3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486` を基にしています。SSIMの専用カーネルはBrushの変更から移植し、v0.3.0のSSIM式と精度に合わせています。[公式タグ](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0) · [SSIMの変更](https://github.com/ArthurBrussee/brush/pull/394)

`training.patch` に、GPUでのSSIM計算、バイナリPLYへの書き出し、学習ログの差分が入っています。版を特定するハッシュと計測条件は[測定記録](../../capture-review/training-speed-v001/README.md)にあります。

## 再ビルド

Windows x64でRust 1.90.0とMSVCを使用しました。次のコマンドでソース取得とリリースビルドを再現できます。

```powershell
git clone --depth 1 --branch v0.3.0 https://github.com/ArthurBrussee/brush.git
Set-Location brush
git apply 'D:/vibe/house-gaussian-splat/apps/brush-0.3.0-training/training.patch'
cargo build --locked --release -p brush-app --bin brush_app
```
