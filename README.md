# Rain Rarity Flood Depth Live

研究・試験用のリアルタイム浸水リスクWebアプリです。

## 主要機能
- ブラウザ現在地 / 地図クリック
- JMA AMeDASの実測雨量（最寄り観測点）を優先取得
- 防災科研「大雨の稀さ」公開ArcGIS ImageServerから地点値を取得
  - 半減期1.5時間実効雨量の稀さ: `e90mrp`
  - 3時間: `r03hrp`
  - 6時間: `r06hrp`
  - 24時間: `r24hrp`
- 国土地理院標高APIから標高と周辺比標高
- 直近雨量時系列から3時間先の保守的な雨量上限をスクリーニング推定
- 実浸水深CSVをアップロードするとGradient Boosting Quantile RegressionでP50/P95浸水深を推定

## 安全上の設計
未学習時は浸水深(m)を捏造せず、危険度のみ表示します。浸水深は実測浸水深データでモデルを学習した後だけ表示されます。

## CSV列
`rain_1h_mm,rain_3h_mm,rain_24h_mm,rain_upper_3h_mm,rarity_e90m_y,rarity_r3h_y,rarity_r6h_y,rarity_r24h_y,elevation_m,relative_elevation_m,depth_m`

## ローカル実行
```bash
pip install -r requirements.txt
python app.py
```
http://localhost:8000

## Railway
GitHubへこのフォルダをpushし、Railwayでリポジトリを接続するとDockerfileから起動できます。`PORT`はRailwayが自動設定します。

## 注意
- `rain_upper_3h_mm` は短時間の統計的スクリーニング値で、気象業務法上の正式な予報ではありません。
- 防災科研/JMA/GSI側の仕様変更や通信障害時は値が欠損する可能性があります。
- 避難判断では気象庁、自治体等の公的情報を優先してください。
