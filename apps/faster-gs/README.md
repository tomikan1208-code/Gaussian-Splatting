# 学習用 Faster-GS

`Train-Room.cmd` の学習方式で `2` を選ぶと Faster-GS、`3` を選ぶと高速版の Faster-GS を使います。どちらも既存のCOLMAPデータを受け取り、向き補正済みの `room.ply` を出力します。高速版は3スレッドで最大6枚の画像を先読みし、同じ画素をCUDAに転送します。カメラ行列とカメラ位置は学習中にGPUから再利用します。

高速版では、カメラが100枚以上あり、撮影中心からの最大距離が99パーセンタイルの2倍を超える場合、学習率と点の増減に使う範囲を99パーセンタイルから求めます。学習には全カメラを使います。 [このPCでの計測](../../capture-review/faster-gs-v001/README.md)

上流の `main` を固定し、Windowsのビルドパス、単一画像のCOLMAPカメラ、点数上限、バイナリPLYの出力をこのプロジェクトに合わせています。PPISPは無効で、初期点の近傍探索はscikit-learnを使います。変更は [Faster-GSの差分](faster-gs.patch) と [NeRFICGの差分](nerficg.patch) にあります。 [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting/tree/4f8e18690464250c2443bfed79d17a36f5c99faf) · [NeRFICG](https://github.com/nerficg-project/nerficg/tree/623ad16ec6e05a5c5cedacc046273b63eac2686e)

## 再セットアップ

Windows x64、CUDA対応NVIDIA GPU、Python 3.11、Git、uv、Visual Studio 2022のx64 C++ビルドツールが必要です。プロジェクト内にCUDA 12.8.1のビルドに必要なファイルを取得し、PyTorch 2.8用にCUDA拡張を作ります。初回は数GBを取得します。 [NVIDIAの配布マニフェスト](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) · [PyTorch 2.8の配布手順](https://pytorch.org/get-started/previous-versions/#v280)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\vibe\house-gaussian-splat\apps\faster-gs\Setup-FasterGS.ps1'
```
