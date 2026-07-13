# P5-E2 Corrective Training Design

## Goal

P5-E2の最終checkpointで確認された次の二つの失敗を、シム真値に依存しないモデルとして修正する。

1. 手先が対象物へ接近できず、把持・配置が成立しない。
2. 物理進捗がないのにpick/placeの`(done)`や`Task completed.`を生成する。

修正は、短いaction horizon、成功デモ500エピソード、失敗状態からのcorrective data
100エピソード、state-randomized text-only学習を組み合わせる。シム真値は教師ラベルと評価にのみ
使用し、推論時のモデル入力やゲートには使用しない。

## Confirmed Findings

- 最終checkpointは未見seedで0/30成功・mean placed 0.0だった。
- 収集に使ったseed 0〜2でも0/3成功・mean placed 0.0であり、未見配置への汎化不足だけでは
  説明できない。
- `previous_narrations`は教師データからLeRobotのcomplementary dataへ伝播し、token化後の
  `Progress:`にも含まれる。履歴非空は38,092/38,642 frame（98.6%）。
- 実況target非空は6,650/38,642 frame（17.2%）。最大prompt長は240 tokenで、固定長256による
  切断はない。
- 最終checkpointは`All keys loaded successfully!`でロードでき、state dict警告はない。
- 既存`debug_inference.py`のaction MSEは50-step予測chunkを現在frameの1-step actionへbroadcast
  比較しており、行動品質の健全性指標として使用できない。
- 実況あり30エピソードでは、物体を配置しないまま手続き的実況だけが最後まで進んだ。

## Architecture

### 1. State-randomized text-only training

`SNVLAConfig`に次の設定を追加する。

```text
state_randomization_text_only_enabled: bool = False
state_randomization_text_only_ratio: float = 0.25
```

この機能は学習時だけ動作する。有効時は各サンプルを設定比率で選び、Normalizer適用後の実state
各次元を独立な`Uniform(-1, 1)`で置換する。画像、task、`previous_narrations`、
`current_narration`は維持する。選択サンプルでは`diffusion_loss_mask=0`とし、action lossを無効化して
text lossだけを計算する。

random stateはtokenizer stepがprompt用の`state_str`を作る箇所でだけ使用し、正規化済みbatchの
`observation.state`自体は書き換えない。`diffusion_loss_mask`はprocessor transitionから最終batchまで
保持されるようconverter境界を明示的に対応させる。

実況あり・なしの双方へ同じ確率で適用し、「ランダムstateなら必ずBON」というshortcutを作らない。
無効時は乱数生成を含めて既存挙動を変更しない。FSDP各rankの乱数は学習seedから再現可能にする。

### 2. Successful demonstrations

成功デモは合計500エピソードとする。

- 既存50エピソードを保持する。
- 未使用seed帯から450エピソードを追加収集する。
- 全エピソードについてexpert success、picked/placedイベント列、実況ストリーム、frame数を検証する。
- 既存と同じforward-only規約で実況を増強し、イベント確定frameより前へ完了実況を伝播しない。

### 3. Corrective data

現checkpointの失敗状態からscripted expertへ引き継ぐcorrective episodeを100本収集する。

1. policyをランダムな介入時点までロールアウトする。
2. policy区間の実況教師はモデル出力ではなくシム真値イベントから再構築する。
3. 物理イベントなしでモデルが実況を進めたframeの教師modeはBOAとする。
4. policy区間は`diffusion_loss_mask=0`とし、失敗actionを学習しない。
5. 同じ環境状態からscripted expertへ切り替えて回復軌道を収集する。
6. expert区間だけaction lossを有効化する。

最初に10エピソードのpilotを行い、scripted expertがpolicyの失敗状態から回復してtaskを完了できる
ことを確認する。pilotが失敗した場合は100本収集へ進まず、expertの任意状態再開を先に診断する。
corrective datasetに保存した`diffusion_loss_mask`がdataloaderとprocessor converterを通過してmodel
`forward`へ届くことを統合テストで確認する。

### 4. Action horizon selection

再学習前に、現checkpointを収集seed上で`n_action_steps=1/5/10/30`として比較する。指標はsuccess
だけでなく、end-effectorから現在対象物までの最小距離、picked数、placed数とする。最長の安定値を
採用し、全構成がsuccess 0でも対象物への接近距離が改善する値を選ぶ。

学習するchunk長は50のままとし、`n_action_steps`は推論時に実行するreceding horizonとして扱う。

### 5. Dataset composition and split

最終データセットは成功デモ500エピソードとcorrective 100エピソードで構成する。episode単位で10%を
validationへ分け、同一episodeのframeがtrainとvalidationへ跨らないようにする。

- 通常の成功デモ: action loss有効
- correctiveのexpert区間: action loss有効
- correctiveのpolicy区間: action loss無効
- state-randomized text-onlyサンプル: action loss無効

## Metrics and W&B

本番学習ではW&B連携を必ず有効にする。run名にはデータセット版、action horizon、state randomization
比率を含める。少なくとも次の値を通常ログとW&Bの両方へ記録する。

- total loss、text loss、action loss
- text/action loss ratio
- action loss有効サンプル数と比率
- state-randomized text-only実適用数と比率
- 実況あり/なし別のBON/BOA mode loss
- 通常/state-randomized別のtext loss
- validationの同一指標
- 正しい50-step教師chunk対予測chunkのMSE/MAE（padding部分を除外）
- learning rate、gradient norm、step time、GPU memory

W&B初期化またはログ送信に失敗した場合は、黙ってW&Bなしで本番学習を続けない。原因を報告して
再開判断を求める。

## Evaluation Metrics

行動系:

- end-effectorから現在対象物までの最小距離
- picked / placed / success
- action horizon別の上記指標
- time indexとpadding maskを揃えたchunk MSE/MAE

実況系:

- pickedイベントなしで生成したpick `(done)` 数
- placedイベントなしで生成したplace `(done)` 数
- 必要placed数未達で生成した`Task completed.`数
- BON/BOA確率と実況列

推論時にこれらの真値をモデルへ戻して実況を抑制してはならない。真値は評価指標にのみ使用する。

## Stage Gates

1. corrective pilot 10本でexpert回復成功を確認する。
2. 小規模学習で分離lossと正しいchunk MSEが改善することを確認する。
3. 収集seed 3/3の各エピソードで少なくとも1回pickedを達成する。
4. 同3エピソードで虚偽`(done)`と早すぎる`Task completed.`を0件にする。
5. 上記を通過したcheckpointだけを未見seedの実況あり/なし各30エピソードで記録付き評価する。
6. 中間checkpointにも同じゲートを適用し、最終stepを自動的に採用しない。

checkpoint破損、`Warning: Could not load state dict`、expert回復失敗、loss不均衡、W&B不成立、または
収集seedゲート不通過では次段階へ進まない。

## Test Strategy

実装はTDDで行い、次の失敗テストを先に作成する。

- state randomizationは設定有効時だけ適用される。
- 比率0.0では適用されず、1.0では全サンプルへ適用される。
- random stateの全要素が`[-1, 1]`内にある。
- randomization対象でもtext loss maskは残り、diffusion lossだけが無効になる。
- 同一batch内の実況あり・なしの双方へ適用できる。
- correctiveのpolicy/expert境界で`diffusion_loss_mask`が切り替わる。
- chunk metricは同じtime indexを比較し、padding部分を除外する。
- 真値イベントなしの生成完了をfalse completionとして数える。
- episode単位splitでframe漏洩がない。
- W&B有効設定が学習エントリポイントへ伝播する。

非simテストに加え、corrective pilot、action horizon比較、収集seedゲートをシム統合テストまたは
再実行可能な診断コマンドとして残す。

## Operational Constraints

- DGXで通常作業に使うGPUは`CUDA_VISIBLE_DEVICES=2,3`のみ。
- `max_state_dim/max_action_dim`は`32/32`。
- ローカルは`.venv/bin/python -m ...`形式だけを使う。
- checkpointロード時に`All keys loaded successfully!`を確認する。
- `Warning: Could not load state dict`が出たら即時中止する。
- 評価は実況あり/なし各30エピソードとも記録を残す。
- 未追跡の`outputs/`はコミットしない。
