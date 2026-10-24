# LeRobotによる双腕模倣学習 教材（箱の蓋閉め）― Meta Quest × DOBOT Magician E6 ×2台

Oct 22, 2026 · @春木

## この教材について

この教材では、Meta Quest 3 / 3S で2台の DOBOT Magician E6 を遠隔操作して「お手本データ」を集め、LeRobot で学習したAIに同じ作業を自律で行わせます。課題は「ロボットBがワークを箱に入れ、ロボットAがノズルを傾けて（Rx）箱の蓋を閉める」双腕作業です。人が手で教え、AIがまねて動く、という模倣学習の一連の流れを実機で体験します。

### ねらい

- 模倣学習の基本要素（観測・行動・エピソード・方策）を、実際のデータを作りながら理解する
- データの質と量が、AIの動きの良し悪しを決めることを体験する
- 「収集 → 学習 → 評価 → 改善」のサイクルを1周回す

### 前提

| 項目 | 内容 |
| --- | --- |
| 完了していること | 「Meta Quest Touch Plus による DOBOT Magician E6 ×2台 遠隔操作 手順書」の全手順（2台が Quest で操作でき、`h` で初期位置に戻れる状態） |
| 作業フォルダ | `C:\quest_dobot`（手順書のプログラム一式がある） |

### 全体の流れと時間の目安

| 時限 | 章 | 内容 | 時間 |
| --- | --- | --- | --- |
| 1 | 第1章〜第2章 | 模倣学習のしくみ、環境構築 | 50分 |
| 2 | 第3章〜第5章 | 作業環境づくり、プログラム追加、動作確認 | 50分 |
| 3〜4 | 第6章〜第7章 | データ収集（50エピソード）と確認 | 100分 |
| ― | 第8章 | Google Colab で学習（待ち時間。授業外でも可） | 約1.5〜3時間 |
| 5 | 第9章 | 学習済みモデルで自律動作 | 50分 |
| 6 | 第10章 | 評価・改善、発表 | 50分 |

この教材で追加するプログラムは、手順書のプログラムをそのまま利用します。手順書側のファイルは `static/index.html` だけを差し替え、それ以外は変更しません。

## 第1章 模倣学習とLeRobotのしくみ

模倣学習は「この画面・この状態のときは、こう動かす」という人の判断を、たくさんの記録からAIに覚えさせる方法です。AIが覚えた判断のルールを\*\*方策（ポリシー）\*\*と呼びます。

### 1-1. 用語

| 用語 | 意味 | この教材では |
| --- | --- | --- |
| 観測（observation） | AIが見るもの・知る状態 | カメラ2台の映像 ＋ 2台の手先位置・Rx・ツール状態（A=DO1、B=吸気） |
| 行動（action） | AIが出す指令 | 2台の次の目標位置・目標 Rx・ツール ON/OFF |
| フレーム | ある瞬間の「観測＋行動」の組 | 1秒間に10フレーム（10 fps）記録 |
| エピソード | 作業1回分（開始〜終了）のフレームの並び | 1回 約20〜40秒 ＝ 200〜400フレーム |
| データセット | エピソードの集まり | 50エピソードを目安に収集 |
| 方策（ポリシー） | 観測から行動を決めるAIモデル | ACT（Action Chunking with Transformers） |

### 1-2. 観測と行動の中身

2台のロボットは、位置（X, Y, Z）に加えて **Rx（ベース座標の X 軸まわりの回転＝ノズルの傾き）** と、トリガーで操作する出力（ロボットA はDO1、ロボットB は吸気）を扱います。Ry・Rz は直接は操作しません。蓋を押して倒すときにノズルを傾けるため、Rx が必要になります。

| データ名 | 形 | 内容 |
| --- | --- | --- |
| `observation.state` | 10個の数値 | `A.x, A.y, A.z, A.rx, A.do1, B.x, B.y, B.z, B.rx, B.suction`（位置は mm、Rx は deg、DO1・吸気 は 0/1） |
| `observation.images.top` | 480×640 カラー画像 | 作業台を真上から撮るカメラ |
| `observation.images.front` | 480×640 カラー画像 | 作業台を正面斜め上から撮るカメラ（蓋の角度が見える） |
| `action` | 10個の数値 | `observation.state` と同じ並びの「次に向かう目標位置・目標 Rx」と「DO1・吸気 の指令」 |

> Rx は「初期姿勢の Rx（`RX_CENTER`、下向きのノズルでは 180° 付近）からのずれ」で記録します。E6 の Rx は 180° と −180° が同じ向きなので、そのまま記録すると 179° → −179° のように値が跳び、AIが学びにくくなるためです。

### 1-3. LeRobotとは

LeRobot は Hugging Face が公開しているロボット学習用のオープンソースライブラリです。データセットの形式（LeRobotDataset）、学習（`lerobot-train`）、主要な方策（ACT、Diffusion Policy など）がそろっています。この教材では LeRobot 0.6 系を使います。

### 1-4. ACTの特徴

ACT は、1回の推論で「この先しばらくの行動」をまとめて予測します（アクションチャンク）。少ないデータでも滑らかな動きを学びやすく、50エピソード程度から実用的な結果が出やすい方策です。この教材では10 fps で 5秒分（50ステップ）を一度に予測し、そのうち最初の1秒分（10ステップ）を実行してから予測し直します。

### 1-5. 全体の流れ

1. **データ収集（PC＋実機）**：Quest で2台を操作し、カメラ映像と位置を1秒10回記録する
2. **学習（Google Colab の GPU）**：データセットを Colab に送り、ACT を学習させる
3. **自律動作（PC＋実機）**：学習済みモデルを PC に戻し、カメラ映像から2台を自動で動かす
4. **評価・改善**：成功率を数え、データを足して学習し直す

> 制御PCにGPUは不要です。学習だけを Google Colab の GPU で行い、自律動作（推論）は PC の CPU で行います。

&#91;embedded content: 模倣学習のサイクル · 4段階と改善ループ\]

データ収集と自律動作は PC と実機で、学習だけを Colab の GPU で行います。評価で見つけた失敗に合わせてデータを足し、もう一度学習します。

## 第2章 器材と環境構築

この教材で使う準備物をまとめます。手順書の器材（ロボット・Quest・PC など）に、USBカメラ2台と作業用の小物を追加します。PC には LeRobot 用に Python 3.12 の仮想環境を作ります。

### 2-1. 準備物

#### ロボット・操作機器（手順書と共通）

| No. | 器材 | 数量 | 備考 |
| --- | --- | --- | --- |
| 1 | DOBOT Magician E6 | 2台 | [DOBOT Magician E6 PythonセットB（AFD-E6PYB）](https://afrel.co.jp/product/dobot-education/lineup/#pythonset-b) で購入できる。ES01 電動サクションカップ・小型電磁グリッパーキット・Python マスターガイド・演習用サンプルワーク（3個）が付属 |
| 2 | 非常停止スイッチ | 2個 | DOBOT Magician E6 に付属。各ロボットに必ず接続し、補助者がすぐ押せる位置に置く |
| 3 | Meta Quest 3 / 3S | 1台 | Touch Plus コントローラ（左右）付き。パススルーで実機を見ながら操作する |
| 4 | Windows PC | 1台 | Windows 10 / 11。Wi-Fi・有線 LAN・USB ポート2つ以上（カメラ用）。GPU は不要（学習は Google Colab で行う） |
| 5 | Wi-Fi ルーター | 1台 | PC と Quest を同じネットワークにつなぐ |
| 6 | LAN ケーブル・スイッチングハブ | 1式 | PC と2台のロボットを有線 LAN でつなぐ（ロボットの IP は 192.168.5.1 / 192.168.5.2） |
| 7 | 小型電磁グリッパー（ロボットA の DO1 につなぐ） | 1個 | DOBOT Magician E6 PythonセットB に付属の小型電磁グリッパーキット（ツメ・フランジを含む）を使える。制御盤の DO1 で ON / OFF する |

> ロボットB の吸気には、E6 に付属の ES01 電動サクションカップを使います。ES01 はツール側の DO1 で動くので、`config.py` のロボットB は `"suction_on": "ToolDOInstant(1,1)"` / `"suction_off": "ToolDOInstant(1,0)"` にします。この設定で吸着しないときは、`python find_output.py B` で探せます。

#### この教材で追加する器材

| No. | 器材 | 数量 | 備考 |
| --- | --- | --- | --- |
| 1 | USBウェブカメラ | 2台 | 640×480・30fps 以上。オートフォーカスは切れる機種が望ましい |
| 2 | カメラスタンド（アーム式） | 2本 | 真上用・正面用。撮影中に動かないよう固定する |
| 3 | LEDライト | 1〜2灯 | 窓の光や影の変化を減らす（学習の安定に重要） |
| 4 | ワーク | 3〜5個 | 吸着できる平らなブロック（一辺 30〜40mm、同じ色・形） |
| 5 | トレイ・箱・マーカー | 1式 | 右トレイ（ワーク置き場）。ビニールテープで作業台に枠を描く。箱は厚紙の箱（約100×100×60mm）に布テープで蝶番をつけた蓋と、蓋を止めるブロック1個 |
| 6 | 作業台の天板 | 1枚 | 無地で、ワークと色がはっきり違うもの |
| 7 | Google アカウント | 1つ | Colab と Google ドライブで使う |

> USBカメラ2台は、PC の別々の USB ポート（できればハブを使わず本体のポート）に接続します。帯域不足で映像が止まるのを防ぐためです。

### 2-2. Python 3.12 のインストール

LeRobot 0.6 は **Python 3.12 以上** が必要です。手順書の Python（3.10 以上）とは別に、仮想環境を作ります。

1. [python.org](https://www.python.org/downloads/windows/) から Python 3.12 の Windows 版をダウンロードしてインストールする（「Add python.exe to PATH」にチェック）
2. コマンドプロンプトでバージョンを確認する

```bat
py -3.12 --version
```

### 2-3. 仮想環境の作成と LeRobot のインストール

```bat
cd C:\quest_dobot
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install "lerobot[dataset]==0.6.1" aiohttp cryptography matplotlib
```

インストールには10分前後かかります。以降、この教材のプログラムは**必ず仮想環境を有効にしてから**実行します。新しいコマンドプロンプトを開いたら、最初に次を実行してください。

```bat
cd C:\quest_dobot
.venv\Scripts\activate
```

プロンプトの先頭に `(.venv)` と表示されていれば有効になっています。

### 2-4. インストールの確認

```bat
python -c "import lerobot, torch, cv2; print(lerobot.__version__, torch.__version__, cv2.__version__)"
```

`0.6.1` と PyTorch・OpenCV のバージョンが表示されれば完了です。

> バージョンを `0.6.1` に固定しているのは、PC と Colab で同じ LeRobot を使うためです。PC と Colab でバージョンが違うと、学習済みモデルを読み込めないことがあります。

## 第3章 作業環境とタスク設計

模倣学習の成否は、作業環境を毎回同じにできるかで大きく変わります。カメラ・照明・トレイの位置を決めたら、データ収集から自律動作まで動かさないことが原則です。

### 3-1. 課題（タスク）：ワークを箱に入れて蓋を閉める

2台が順番に協力して、ワークを箱にしまい、蓋を閉めます。

1. ロボットB（右コントローラ）が、右トレイのワークを吸気で吸着して持ち上げ、箱の中に下ろして放し、上へ退避する
2. ロボットA（左コントローラ）が、開いている蓋の外側（左側）に近づき、**手首をひねってノズルを傾け（Rx）**、蓋の上のほうに当てる
3. ロボットAが蓋を右へ押して倒し、閉じた蓋の上を軽く押さえる
4. ロボットAが上へ退避し、Rx を元の角度に戻して終了

この課題の見どころは2つです。ロボットAが「Bが退避するまで待つ」タイミングと、蓋の角度に合わせて「ノズルを傾ける」動きを、AIがカメラ映像から学べるかどうかです。

### 3-2. トリガーで操作する出力

ロボットAは小型電磁グリッパーをつないだ **DO1** を、ロボットBは吸気を操作します。2台とも、トリガーを引くたびに ON / OFF が切り替わります。

| ロボット | トリガーで操作する出力 | 動き |
| --- | --- | --- |
| A（左コントローラ） | DO1（制御盤のデジタル出力1） | トリガーを引くたびに ON / OFF を切り替え |
| B（右コントローラ） | 吸気 | **トリガーを引くたびに ON / OFF を切り替え（手順書と同じ）** |

ロボットA の DO1 をどの場面で ON / OFF にするかは、接続した機器に合わせてグループで決め、毎回同じタイミングで操作します。DO1 の ON/OFF も記録され、AIがタイミングを学習します。ツール側の DO1 を使う場合は、`lerobot_config.py` の `TOOLS` でコマンドを `ToolDOInstant(1,1)` / `ToolDOInstant(1,0)` に変えます。

> プログラムの起動時と終了時は、ロボットAの DO1 とロボットBの吸気はどちらも OFF になります。DO1 につなぐ機器がこの動きで安全か、事前に確認してください。

### 3-3. 箱と蓋の作り方

| 項目 | 仕様 |
| --- | --- |
| 箱 | 厚紙の箱（約 100×100×60mm、内側にワークが入る大きさ）。両面テープで作業台に固定する |
| 蓋 | 箱の**左側（ロボットA側）の辺**を布テープで蝶番にする。蝶番の軸はロボットの X 方向（前後）に平行 |
| 開いた角度 | 約 100°（垂直より少し外側に倒れた状態）。外側にブロックを置いて止め、毎回同じ角度にする |
| 閉じ方 | 蓋の上のほうを右へ押して垂直を越えれば、自重で倒れて閉じる |
| ワーク | 吸着できる平らなブロック（一辺 30〜40mm）。箱の中に入る大きさ |

&#91;embedded content: 作業台の配置例 · 真上から見た図\]

操作者はロボットの後ろに立ち、ロボットと同じ向き（X+ 方向）を向いて操作します。ロボットAは蓋の外側（左）から近づき、ノズルを右へ傾けて蓋を押します。

#### 箱の置き方（手順）

1. 手順書の `quest_dobot_teleop.py` を起動し、`h` で2台を初期位置へ移動させる
2. 箱を **2台のちょうど中間** に置く。前後（X 方向）は、2台のノズルが箱の上まで届く位置にする
3. 蓋の蝶番を **左側（ロボットA側）** にして、蓋を左へ開く。蝶番の軸はロボットの X 方向（前後）と平行にする
4. 蓋の外側に止めブロックを置き、開いた角度を約 100°（垂直より少し外側に倒れた状態）にそろえる
5. 箱の底を両面テープで作業台に固定し、四隅をビニールテープで印をつける（ずれたときに戻せるように）
6. 右トレイを箱の **右側（ロボットB側）** に置き、ビニールテープで枠を描く
7. ゆっくり遠隔操作して、ロボットB のノズルが右トレイの中と箱の中央の上に届くこと、ロボットA のノズルが開いた蓋の外側（左）に届き傾けても箱や作業台に当たらないこと、箱の上で2台がぶつからないことを確かめる（ぶつかる場合は `config.py` の `limits` の Y 範囲を狭める）
8. 真上カメラに2台のノズル・箱・右トレイがすべて写り、正面カメラに蓋の角度が写ることを確認する

> 記録を始めたら、箱は動かさないでください。箱の位置が変わると、AIが覚えた位置とずれて失敗します。動いてしまったときは、テープの印に合わせて戻します。

### 3-4. 配置のルール

| 項目 | ルール |
| --- | --- |
| ロボットの間隔 | 箱の上に2台とも届くこと。`limits` の Y 範囲で、Bは箱の右半分まで、Aは箱の左外側〜箱の中央までに制限する（手順書 7-2） |
| 真上カメラ（top） | 作業台の中央の真上 60〜80cm。2台のノズル・箱・右トレイがすべて写る |
| 正面カメラ（front） | 操作者側から斜め45°程度。**蓋の角度とノズルの傾き**が見える角度 |
| 照明 | LEDライトを固定し、カーテンを閉める。影が毎回同じになるようにする |
| 箱・トレイ | 箱は両面テープで固定。右トレイはビニールテープで枠を描き、毎回同じ場所に置く |
| 背景 | カメラに人の手や服が写り込まない位置に操作者と補助者が立つ |

### 3-5. 置き方に「ばらつき」をつける

AIが「いつも同じ場所」しか覚えないと、少しずれただけで失敗します。箱は固定したまま、ワークの位置と蓋の開き角をエピソードごとに少しずつ変えます。

| エピソード番号 | 右トレイ内のワーク位置 | 蓋の開き角 |
| --- | --- | --- |
| 0〜9 | 枠の中央 | 約 100° |
| 10〜19 | 枠の左寄り・右寄りを交互 | 約 100° |
| 20〜29 | 枠の手前寄り・奥寄りを交互 | 約 100° |
| 30〜39 | 枠の中央 | 約 95°・110° を交互（止めブロックの位置を変える） |
| 40〜49 | 枠内のランダムな位置 | 約 95〜110° のランダム |

### 3-6. 自己チェック

- [ ] 真上カメラに、2台のノズル・箱・右トレイがすべて写っている
- [ ] 正面カメラで、蓋の角度とノズルの傾きが見分けられる
- [ ] 箱を固定し、蓋が止めブロックで毎回同じ角度に開く
- [ ] 蓋の上のほうを指で右へ押すと、自重で倒れて閉じる
- [ ] カメラ・ライト・トレイをテープで固定し、位置に印をつけた
- [ ] 2台の動作範囲（limits）で、箱の上で2台がぶつからないことを、ゆっくり操作して確認した

## 第4章 プログラム

手順書のフォルダ `C:\quest_dobot` に、模倣学習用の Python ファイル9つとノートブックを追加し、`static\index.html` を差し替えます。手順書の `config.py`・`dobot_client.py`・`quest_dobot_teleop.py` はそのまま部品として使います。

### 4-1. フォルダ構成

```
C:\quest_dobot\
 ├─ config.py                 ← 手順書（IP・初期位置・動作範囲・吸気）※変更なし
 ├─ dobot_client.py           ← 手順書（E6 通信）※変更なし
 ├─ quest_dobot_teleop.py     ← 手順書（遠隔操作）※変更なし
 ├─ get_pose.py               ← 手順書（現在位置の確認）※変更なし
 ├─ cert.pem / key.pem        ← 手順書で作成済み
 ├─ static\index.html         ← ★差し替え（コントローラの向き・A/B/X/Y ボタンの送信を追加）
 │
 ├─ lerobot_config.py         ← ★追加：模倣学習の設定（Rx の設定を含む）
 ├─ camera.py                 ← ★追加：USBカメラの読み取り
 ├─ find_cameras.py           ← ★追加：カメラ番号の確認
 ├─ e6_common.py              ← ★追加：共通部品（記録項目・キー入力・電源確認・自動運転）
 ├─ teleop_rx.py              ← ★追加：遠隔操作に Rx（手首ひねり）を追加
 ├─ record_dataset.py         ← ★追加：遠隔操作しながらデータ収集
 ├─ check_dataset.py          ← ★追加：データセットの確認
 ├─ replay_episode.py         ← ★追加：記録した動きの再生
 ├─ run_policy.py             ← ★追加：学習済みモデルで自律動作
 ├─ train_act_colab.ipynb     ← ★追加：Colab 学習用ノートブック
 │
 ├─ .venv\                    ← 第2章で作成した仮想環境
 ├─ data\e6_box_lid\          ← 記録したデータセット（自動で作成）
 └─ models\act_e6\pretrained_model\  ← 学習済みモデル（第8章で配置）
```

各ファイルは UTF-8 で保存してください。全ファイルは本教材の配布フォルダにも同梱しています。

### 4-2. `lerobot_config.py`（模倣学習の設定）

```python
# =====================================================================
#  lerobot_config.py
#  模倣学習（LeRobot）用の設定ファイル
#
#  ・ロボットの IP・動作範囲（X,Y,Z）・吸気コマンドなどは、手順書の config.py を
#    そのまま使います（このファイルには書きません）
#  ・ここには「データセット」「カメラ」「Rx（手首ひねり）」「ツール（DO1・吸気）」「自律動作」の設定を書きます
# =====================================================================

import math                    # 姿勢（回転）の計算
import os

# このファイルがあるフォルダ（C:\quest_dobot）
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
#  データセットの設定
# ---------------------------------------------------------------------
# データセットの名前（「所有者/名前」の形。ローカル保存のみなら所有者は local でよい）
DATASET_REPO_ID = "local/e6_box_lid"

# データセットの保存先フォルダ（C:\quest_dobot\data\e6_box_lid）
DATASET_ROOT = os.path.join(BASE, "data", "e6_box_lid")

# 作業内容の説明文（データセットに記録される。学習・推論で同じ文を使う）
TASK = ("Robot B puts the block from the right tray into the box, "
        "then robot A tilts its tool and closes the box lid.")

# ロボットの種類名（データセットのメタ情報として記録される）
ROBOT_TYPE = "dobot_e6_bimanual"

FPS = 10                     # 1秒あたりの記録回数 [フレーム/秒]
EPISODE_TIME_SEC = 60        # 1エピソードの最大時間 [秒]（超えると自動で保存して終了）

# ---------------------------------------------------------------------
#  カメラの設定（find_cameras.py で番号を確認して書き換える）
#    index : カメラ番号（0, 1, 2 …）
#    width, height : 解像度 [ピクセル]
# ---------------------------------------------------------------------
CAMERAS = {
    "top":   {"index": 0, "width": 640, "height": 480},   # 真上から作業台全体
    "front": {"index": 1, "width": 640, "height": 480},   # 正面斜め上から（蓋の角度が見える位置）
}

# ---------------------------------------------------------------------
#  Rx（ベース座標の X 軸まわりの回転＝手首ひねり）の設定
#    手首をひねると、ノズルがロボットのベース座標の X 軸まわりに回る
#    （ティーチングペンダントの「ベース座標で Rx をジョグ」と同じ回り方）
#    ※ 姿勢の数値 Rx だけを変えると、ツールの向きによっては Ry 方向に回って見えるため、
#      回転行列で「ベースの X 軸まわり」に回した姿勢を計算して送る
#
#    ROT_REF    : 基準の姿勢 [Rx, Ry, Rz] [deg]。初期位置で get_pose.py を実行し、
#                 「ROT_REF 用」に表示された値を書く。
#                 None のままだと、プログラム起動時のロボットの姿勢を基準にする
#                 （記録と自律動作で基準がずれないよう、値を書いておくのがおすすめ）
#    RX_RANGE   : 基準の姿勢から左右に何度まで傾けてよいか [deg]
#    RX_SIGN    : ひねる向きとロボットの回転が逆なら -1 にする
#    RX_SCALE   : 手首のひねり角に対するロボットの回転の倍率（0.5 で半分の角度）
#    RX_MAX_SPEED : Rx の最大回転速度 [deg/s]
# ---------------------------------------------------------------------
ROT_REF = {"A": None, "B": None}     # 例: {"A": [180.0, 0.0, 90.0], "B": [180.0, 0.0, 90.0]}
RX_RANGE = 40.0
RX_SIGN = {"A": 1, "B": 1}
RX_SCALE = 1.0
RX_MAX_SPEED = 45.0


# ---- 姿勢の計算（DOBOT の Rx, Ry, Rz は X→Y→Z の順に固定軸まわりに回した角度） ----
def euler_to_mat(rx, ry, rz):
    """姿勢 Rx, Ry, Rz [deg] → 回転行列 R = Rz・Ry・Rx（3×3 のリスト）"""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return [[cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
            [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
            [-sb,     cb * sa,                cb * ca]]


def mat_to_euler(R):
    """回転行列 → 姿勢 Rx, Ry, Rz [deg]（euler_to_mat の逆）"""
    sy = math.hypot(R[0][0], R[1][0])                 # cos(Ry) の大きさ
    if sy > 1e-6:                                     # ふつうの場合
        rx = math.atan2(R[2][1], R[2][2])
        ry = math.atan2(-R[2][0], sy)
        rz = math.atan2(R[1][0], R[0][0])
    else:                                             # Ry = ±90°（特異姿勢）の場合
        rx = math.atan2(-R[1][2], R[1][1])
        ry = math.atan2(-R[2][0], sy)
        rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def mat_mul(A, B):
    """3×3 行列の積 A・B"""
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_t(A):
    """3×3 行列の転置（回転行列では逆回転になる）"""
    return [[A[j][i] for j in range(3)] for i in range(3)]


def rot_x(deg):
    """ベースの X 軸まわりに deg [deg] 回す回転行列"""
    a = math.radians(deg)
    return [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]


def rx_angle(pose, ref):
    """
    姿勢 pose [X,Y,Z,Rx,Ry,Rz] が、基準姿勢 ref [Rx,Ry,Rz] から
    ベースの X 軸まわりに何度回っているか [deg] を返す（記録する Rx の値）
    """
    D = mat_mul(euler_to_mat(*pose[3:6]), mat_t(euler_to_mat(*ref)))   # 基準からの回転（ベース座標）
    return math.degrees(math.atan2(D[2][1] - D[1][2], D[1][1] + D[2][2]))   # X 軸まわりの成分


def rotate_x(rot0, deg):
    """姿勢 rot0 [Rx,Ry,Rz] を、ベースの X 軸まわりに deg [deg] 回した姿勢 [Rx,Ry,Rz] を返す"""
    return mat_to_euler(mat_mul(rot_x(deg), euler_to_mat(*rot0)))


def rot_ref(name, pose_now):
    """ロボット name の基準姿勢 [Rx,Ry,Rz]（ROT_REF が None なら今の姿勢を基準にして表示する）"""
    ref = ROT_REF.get(name)
    if ref is None:
        ref = list(pose_now[3:6])
        print(f"[{name}] Rx の基準姿勢 = [{ref[0]:.1f}, {ref[1]:.1f}, {ref[2]:.1f}]（今の姿勢）。"
              f"毎回同じ基準にするには lerobot_config.py の ROT_REF に書いてください")
    return list(ref)


# ---------------------------------------------------------------------
#  ツール（トリガーで操作する出力）の設定
#    ロボットごとに、トリガーで何を操作するかを決める
#      name    : データセットでの名前（A.do1 / B.suction など）
#      mode    : "hold_off" … 普段は ON、トリガーを引いている間だけ OFF
#                "toggle"   … トリガーを引くたびに ON / OFF を切り替え（手順書と同じ）
#      on / off: ON・OFF にするときにロボットへ送るコマンド
#                None のときは手順書 config.py の suction_on / suction_off を使う
#    ロボットA：DO1 … トリガーを引くたびに ON / OFF
#              （制御盤の DO1。ツール側の DO1 なら ToolDOInstant(1,1) / (1,0) に変える）
#    ロボットB：吸気 … トリガーを引くたびに ON / OFF（手順書と同じ。config.py の suction_on / suction_off）
# ---------------------------------------------------------------------
TOOLS = {
    "A": {"name": "do1", "mode": "toggle", "on": "DOInstant(1,1)", "off": "DOInstant(1,0)"},
    "B": {"name": "suction", "mode": "toggle", "on": None, "off": None},
}


def tool_commands(cfg):
    """ロボット cfg のツールの (ON コマンド, OFF コマンド) を返す"""
    t = TOOLS[cfg["name"]]
    on = t["on"] or cfg["suction_on"]                # 指定がなければ手順書の吸気コマンド
    off = t["off"] or cfg["suction_off"]
    return on, off


def tool_default(name):
    """ツールの初期状態（hold_off は ON、toggle は OFF から始める）"""
    return TOOLS[name]["mode"] == "hold_off"


# ---------------------------------------------------------------------
#  状態（state）と行動（action）の並び
#    2台 × (X, Y, Z, Rx, ツール) = 10個の数値
#      ロボットA のツール = DO1 [0/1]、ロボットB のツール = 吸気 [0/1]
#    Rx は基準姿勢（ROT_REF）からベースの X 軸まわりに回った角度 [deg] で記録する
#    この順番は記録・学習・推論で絶対に変えないこと
# ---------------------------------------------------------------------
ROBOT_NAMES = ["A", "B"]     # config.ROBOTS と同じ順番
N_PER_ROBOT = 5              # 1台あたりの数（X, Y, Z, Rx, ツール）
STATE_NAMES = [f"{r}.{k}" for r in ROBOT_NAMES
               for k in ("x", "y", "z", "rx", TOOLS[r]["name"])]
# → ["A.x", "A.y", "A.z", "A.rx", "A.do1", "B.x", "B.y", "B.z", "B.rx", "B.suction"]

# ---------------------------------------------------------------------
#  自律動作（run_policy.py）・再生（replay_episode.py）の設定
# ---------------------------------------------------------------------
# Colab で学習したモデルのフォルダ（pretrained_model フォルダを指定する）
POLICY_PATH = os.path.join(BASE, "models", "act_e6", "pretrained_model")

POLICY_DEVICE = "cpu"        # 推論に使う装置（GPU のない PC は "cpu"）
N_ACTION_STEPS = 10          # 1回の予測で実行する行動の数（10 = 1秒分）。小さいほど頻繁に予測し直す
RUN_TIME_SEC = 60            # 自律動作の制限時間 [秒]
AUTO_MAX_SPEED = 80.0        # 自律動作・再生時の最大速度 [mm/s]（遠隔操作より遅めにする）
AUTO_RX_MAX_SPEED = 30.0     # 自律動作・再生時の Rx 最大回転速度 [deg/s]
SUCTION_THRESHOLD = 0.5      # AI のツール出力（DO1・吸気）がこの値以上なら ON


def wrap_deg(a):
    """角度を -180〜180 [deg] の範囲に直す"""
    return (a + 180.0) % 360.0 - 180.0
```

### 4-3. `camera.py`（USBカメラの読み取り）

```python
# =====================================================================
#  camera.py
#  USB カメラから最新の画像を取り込み続けるクラス
#
#  ・カメラ1台につき1つのスレッドで読み続け、常に「最新の1枚」を保持する
#  ・記録や推論のプログラムは read() でいつでも最新画像を受け取れる
#  ・画像は RGB 順（LeRobot の形式）、形は (高さ, 幅, 3) の uint8
# =====================================================================

import os
import threading
import time

import cv2


class Camera(threading.Thread):
    def __init__(self, name, index, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.name = name            # カメラの名前（"top" など）
        self.index = index          # カメラ番号
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None           # 最新の画像（RGB）
        self.stamp = 0.0            # 最新画像を受け取った時刻
        self.lock = threading.Lock()
        self.running = False
        self.cap = None

    def open(self):
        """カメラを開いて読み取りスレッドを開始する"""
        # Windows では DirectShow を使うと起動が速く安定する
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"カメラ {self.name}（番号 {self.index}）を開けません")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.running = True
        self.start()
        # 最初の1枚が届くまで最大3秒待つ
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError(f"カメラ {self.name} から画像が届きません")
        print(f"[camera {self.name}] 番号 {self.index} を開きました {self.frame.shape[1]}x{self.frame.shape[0]}")

    def run(self):
        """スレッド本体：画像を読み続けて最新の1枚を保存する"""
        while self.running:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # 指定解像度と違う場合はそろえる（学習データの形を一定にするため）
            if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
                bgr = cv2.resize(bgr, (self.width, self.height))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # OpenCV は BGR 順なので RGB に変換
            with self.lock:
                self.frame = rgb
                self.stamp = time.time()

    def read(self, max_age=0.5):
        """最新の画像を返す。max_age 秒より古い画像しかない場合はエラー"""
        with self.lock:
            frame, stamp = self.frame, self.stamp
        if frame is None or time.time() - stamp > max_age:
            raise RuntimeError(f"カメラ {self.name} の画像が更新されていません")
        return frame.copy()

    def close(self):
        self.running = False
        time.sleep(0.1)
        if self.cap is not None:
            self.cap.release()


def open_cameras(cam_cfg):
    """lerobot_config.CAMERAS の内容から全カメラを開いて辞書で返す"""
    cams = {}
    for name, c in cam_cfg.items():
        cam = Camera(name, c["index"], c["width"], c["height"])
        cam.open()
        cams[name] = cam
    return cams
```

### 4-4. `find_cameras.py`（カメラ番号の確認）

```python
# =====================================================================
#  find_cameras.py
#  PC につながっているカメラの番号を調べ、各カメラの画像を保存する
#
#  ・使い方： python find_cameras.py
#  ・camera_check フォルダに cam0.jpg, cam1.jpg … が保存されるので、
#    画像を見て「真上カメラ」「正面カメラ」の番号を lerobot_config.py に書く
#  ・番号 0 はノートPC内蔵カメラのことが多い
# =====================================================================

import os

import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_check")
os.makedirs(OUT, exist_ok=True)
backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

found = 0
for i in range(6):                         # 番号 0〜5 を順に試す
    cap = cv2.VideoCapture(i, backend)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):                    # 露出が安定するまで数枚読み捨てる
        ok, img = cap.read()
    cap.release()
    if not ok:
        continue
    path = os.path.join(OUT, f"cam{i}.jpg")
    cv2.imwrite(path, img)
    print(f"カメラ番号 {i}: {img.shape[1]}x{img.shape[0]}  → {path}")
    found += 1

if found == 0:
    print("カメラが見つかりません。USB の接続と、他のアプリがカメラを使っていないか確認してください")
```

### 4-5. `e6_common.py`（共通部品）

```python
# =====================================================================
#  e6_common.py
#  記録・再生・自律動作で共通に使う部品
#
#   ・dataset_features() : データセットに記録する項目（形・名前）の定義
#   ・KeyReader          : キーボードの1文字入力をスレッドで受け取る（画面クリックで止まらないよう簡易編集モードを切る）
#   ・power_on_if_needed : 電源 OFF なら PowerOn して待つ
#   ・ArmDriver          : 目標位置・Rx へ速度制限つきで滑らかに動かすスレッド
#                          （再生と自律動作で使う。遠隔操作と同じ考え方）
# =====================================================================

import os                            # OS の種類（Windows かどうか）の判定
import queue                         # キー入力をためておく待ち行列
import sys                           # 標準入力（Windows 以外のキー入力）
import threading                     # スレッド（並行して動く処理）
import time                          # 待ち時間・周期の調整

import config                         # 手順書の設定（IP・動作範囲・吸気コマンド）
import lerobot_config as lc           # 模倣学習用の設定
from dobot_client import DobotClient  # 手順書の E6 通信クラス


def clamp(v, lo, hi):
    """v を lo 以上 hi 以下に収める"""
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------
#  電源投入の確認（ROS 2 版の手順書と同じ注意点）
#   ・ロボットの電源が OFF（RobotMode = 3）なら PowerOn() を送る
#   ・PowerOn の直後に EnableRobot を送っても有効にならないため、約20秒待つ
#   ・Docker の ROS 2 bringup が動いていると 29999 番に接続できないので止めておく
# ---------------------------------------------------------------------
def power_on_if_needed(cfg, wait_sec=20):
    bot = DobotClient(cfg["name"], cfg["ip"])            # 確認用に一時的に接続する
    try:
        bot.connect()
    except OSError as e:                                # 接続できない＝他のソフトが使用中など
        raise RuntimeError(
            f"ロボット{cfg['name']}（{cfg['ip']}）に接続できません: {e}\n"
            "  ・DobotStudio Pro や ROS 2（Docker）の bringup が接続中なら終了してください\n"
            "  ・リモート制御モードが TCP/IP か確認してください") from e
    try:
        mode = bot.robot_mode()                         # 現在の状態（5=待機, 9=エラー など）
        if mode == 3:                                   # 3 = 電源 OFF
            print(f"[{cfg['name']}] 電源を入れます（{wait_sec} 秒待ちます）")
            bot.send("PowerOn()")                       # 電源 ON
            time.sleep(wait_sec)                        # 起動が終わるまで待つ
        elif mode == 9:
            print(f"[{cfg['name']}] エラー状態です。この後 ClearError を送ります")
    finally:
        bot.close()                                     # 確認用の接続を閉じる


# ---------------------------------------------------------------------
#  データセットの項目定義
# ---------------------------------------------------------------------
def dataset_features():
    """LeRobotDataset.create() に渡す features を作る"""
    n = len(lc.STATE_NAMES)                             # 数値の数（2台 × 5 = 10）
    feats = {
        # 観測：2台の手先位置 X,Y,Z [mm]・Rx [deg]・ツール状態 [0/1]（A=DO1, B=吸気）
        "observation.state": {"dtype": "float32", "shape": (n,), "names": lc.STATE_NAMES},
        # 行動：2台の目標 X,Y,Z [mm]・目標 Rx のずれ [deg]・ツール指令 [0/1]（state と同じ並び）
        "action": {"dtype": "float32", "shape": (n,), "names": lc.STATE_NAMES},
    }
    # カメラ画像（動画として保存）
    for name, c in lc.CAMERAS.items():
        feats[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (c["height"], c["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return feats


# ---------------------------------------------------------------------
#  キーボード入力（コマンドプロンプトがアクティブなときだけ有効）
# ---------------------------------------------------------------------
def disable_quickedit():
    """
    Windows のコマンド画面の「簡易編集モード」を切る。
    このモードのまま画面をクリックすると「選択」状態になり、表示もキー入力も止まってしまうため。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)                       # 標準入力（キーボード）のハンドル
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):   # 今の設定を読む
            # 0x40 = 簡易編集モード（切る）、0x80 = 拡張フラグ（これを付けないと切れない）
            k32.SetConsoleMode(h, (mode.value | 0x80) & ~0x40)
    except Exception:
        pass                                            # 切れなくても動作には影響しない


class KeyReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)                   # プログラム終了時に自動で止まるスレッド
        self.q = queue.Queue()                          # 押されたキーをためる
        disable_quickedit()                             # 画面クリックで止まらないようにする

    def run(self):
        if os.name == "nt":                             # Windows：Enter なしで1文字ずつ読める
            import msvcrt
            while True:
                ch = msvcrt.getwch()                    # キーが押されるまで待つ
                self.q.put(" " if ch == " " else ch.lower())   # 大文字は小文字にそろえる
        else:                                   # Windows 以外は Enter が必要
            for line in sys.stdin:
                self.q.put(line[:1].lower() if line.strip() else " ")

    def get(self):
        """押されたキーを1つ返す（なければ None）"""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def wait_enter(self):
        """Enter（またはスペース）が押されるまで待つ。q が押されたら False を返す"""
        while True:
            k = self.q.get()                            # キーが押されるまで待つ
            if k in ("\r", "\n", " "):                  # Enter（Windows は \r）・スペース
                return True
            if k == "q":
                return False


# ---------------------------------------------------------------------
#  ArmDriver：目標位置・目標 Rx に向かって速度制限つきで ServoP を送り続ける
# ---------------------------------------------------------------------
class ArmDriver(threading.Thread):
    """
    1台のロボットを「目標 X,Y,Z,Rx」に向けて動かすスレッド。
    set_target() で目標を渡すと、CONTROL_HZ の周期で少しずつ近づける。
    ・動作範囲（limits / RX_RANGE）の外の目標は範囲内に丸める
    ・1周期の移動量を max_speed [mm/s]、回転量を rx_speed [deg/s] で制限する
    ・Rx は基準姿勢（ROT_REF）からベースの X 軸まわりに回った角度 [deg] で扱う（記録データと同じ）
    ・ロボットへは「動作開始時の姿勢をベースの X 軸まわりに回した姿勢」を送る
    """

    def __init__(self, cfg, max_speed, rx_speed):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.name = cfg["name"]
        self.bot = DobotClient(cfg["name"], cfg["ip"])
        self.max_speed = max_speed
        self.rx_speed = rx_speed
        self.ref = None              # Rx の基準姿勢 [Rx,Ry,Rz]（setup で決める）
        self.rx0 = 0.0               # 動作開始時の Rx [deg]
        self.running = True
        self.active = False          # True の間だけ ServoP を送る
        self.lock = threading.Lock()
        self.r0 = None               # 動作開始時の姿勢 [X,Y,Z,Rx,Ry,Rz]
        self.cmd = None              # 現在の指令 [X,Y,Z,Rx]
        self.target = None           # 目標 [X,Y,Z,Rx]
        # ツール（A=DO1 / B=吸気）。suction は「ツールが ON か」を表す
        self.tool_on, self.tool_off = lc.tool_commands(cfg)   # ON / OFF のコマンド
        self.suction = False
        self.want_suction = lc.tool_default(cfg["name"])     # 初期状態（toggle は OFF）

    # ---- 準備・終了 ----
    def setup(self):
        """接続してイネーブルし、制御スレッドを開始する（まだ動かない）"""
        self.bot.connect()                                   # TCP 29999 番に接続
        self.bot.enable()                                    # ClearError → EnableRobot
        self.bot.send(f"SpeedFactor({config.SPEED_FACTOR})")  # 初期位置移動などの速度 [%]
        self._set_suction(self.want_suction)                 # ツールを初期状態に
        self.ref = lc.rot_ref(self.name, self.bot.get_pose())   # Rx の基準姿勢
        self.start()                                         # run() を別スレッドで開始

    def shutdown(self):
        """停止・ツール OFF・切断"""
        self.pause()                                         # ServoP の送信をやめる
        self.running = False                                 # スレッドを終了させる
        try:
            self._set_suction(False)
        except Exception:
            pass
        self.bot.close()

    def _set_suction(self, on):
        """ツール（A=DO1 / B=吸気）の ON/OFF コマンドを送る（lerobot_config.TOOLS）"""
        self.bot.send(self.tool_on if on else self.tool_off)
        self.suction = on

    # ---- 状態 ----
    def read_state(self):
        """[X, Y, Z, Rx, ツール(0/1)] を返す（記録データと同じ並び）"""
        pose = self.bot.get_pose()                           # [X,Y,Z,Rx,Ry,Rz]
        rx = lc.rx_angle(pose, self.ref)                     # 基準姿勢から X 軸まわりに回った角度
        return [pose[0], pose[1], pose[2], rx, 1.0 if self.suction else 0.0]

    # ---- 動作の開始・停止 ----
    def begin(self):
        """現在の姿勢から追従を開始する"""
        pose = self.bot.get_pose()
        rx = lc.rx_angle(pose, self.ref)                     # 今の Rx（基準姿勢からの角度）
        with self.lock:                                      # run() と同時に書き換えないようにする
            self.r0 = pose                                   # この姿勢を X 軸まわりに回して送る
            self.rx0 = rx
            self.cmd = [pose[0], pose[1], pose[2], rx]
            self.target = list(self.cmd)
            self.active = True

    def pause(self):
        """追従を止める（その場で停止）"""
        with self.lock:
            self.active = False

    def set_target(self, xyz, rx, suction):
        """目標 X,Y,Z [mm]・Rx [deg]・ツール ON/OFF を設定する（範囲外は範囲内に丸める）"""
        lim = self.cfg["limits"]                             # 手順書 config.py の動作範囲
        t = [clamp(xyz[0], *lim["x"]), clamp(xyz[1], *lim["y"]), clamp(xyz[2], *lim["z"]),
             clamp(rx, -lc.RX_RANGE, lc.RX_RANGE)]
        with self.lock:
            self.target = t
            self.want_suction = bool(suction)

    def go_home(self):
        """初期位置へ移動（完了まで待つ）"""
        self.pause()                                         # ServoP を止めてから
        time.sleep(0.3)                                      # 少し待って MovJ に切り替える
        hj = self.cfg.get("home_joint")                      # 手順書 config.py の初期位置（関節角）
        if not hj:
            print(f"[{self.name}] home_joint が未設定です")
            return False
        self.bot.movj_joint(hj)                              # 関節角で初期位置へ移動
        return self.bot.wait_idle(timeout=30)                # 到着まで待つ

    # ---- スレッド本体 ----
    def run(self):
        dt = 1.0 / config.CONTROL_HZ                         # 1周期の時間（20Hz なら 0.05 秒）
        while self.running:
            t0 = time.time()
            try:
                with self.lock:                                  # 目標を読む間は書き換えさせない
                    active = self.active
                    want = self.want_suction
                    if active:                                   # 目標へ少しずつ近づける
                        lim_step = [self.max_speed * dt] * 3 + [self.rx_speed * dt]   # 1周期で動ける量
                        for i in range(4):
                            self.cmd[i] += clamp(self.target[i] - self.cmd[i], -lim_step[i], lim_step[i])
                        # 開始時の姿勢をベースの X 軸まわりに (Rx指令 - 開始時のRx) 度回した姿勢
                        rot = lc.rotate_x(self.r0[3:6], self.cmd[3] - self.rx0)
                        pose = self.cmd[:3] + rot
                if active:
                    self.bot.servo_p(pose)                       # 指令位置へ滑らかに移動
                    if want != self.suction:                             # ツールが変わったときだけ送る
                        self._set_suction(want)
            except Exception as e:                               # 通信エラーなど → 停止して待つ
                print(f"[{self.name}] エラー: {e}")
                self.pause()
                time.sleep(0.5)
            time.sleep(max(0.0, dt - (time.time() - t0)))   # 周期を一定に保つ
```

### 4-6. `teleop_rx.py`（遠隔操作に Rx を追加）

手順書の `RobotTeleop` を引き継ぎ、手首のひねりで Rx を回す処理を加えたクラスです。グリップを握った瞬間のコントローラの向きを基準にし、そこから「前向きの軸」まわりにひねった角度だけ Rx を回します。

```python
# =====================================================================
#  teleop_rx.py
#  手順書の遠隔操作（RobotTeleop）に「Rx の制御（手首ひねり）」を追加したクラス
#
#  ・位置 X, Y, Z の動かし方は手順書と同じ（グリップを握っている間だけ追従）
#  ・グリップを握った瞬間のコントローラの向きを基準にして、
#    コントローラを「指している方向の軸」まわりにひねった角度だけ、
#    ノズルをロボットのベース座標の X 軸まわり（Rx）に回す
#    （Quest 側の index.html が targetRaySpace の向きを送る。-Z が指している方向）
#      右へひねる（右手首を時計回り、後ろから見て）→ Rx＋（RX_SIGN=1 のとき）
#  ・姿勢の数値 Rx だけを書き換えると、ツールの向きによっては Ry 方向に回って見えるため、
#    「握った瞬間の姿勢をベースの X 軸まわりに回した姿勢」を計算して Rx, Ry, Rz を送る
#  ・Rx は基準姿勢（ROT_REF）± RX_RANGE の範囲に制限し、RX_MAX_SPEED で速度を制限する
#  ・トリガーで操作するツールをロボットごとに変える（lerobot_config.TOOLS）
#      ロボットA：DO1  … トリガーを引くたびに ON / OFF
#      ロボットB：吸気 … トリガーを引くたびに ON / OFF（手順書と同じ）
# =====================================================================

import math                           # 角度の計算（atan2 など）

import config                         # 手順書の設定（倍率・速度・動作範囲）
import lerobot_config as lc           # Rx の設定（ROT_REF, RX_RANGE など）と姿勢の計算
import quest_dobot_teleop as qt       # 手順書の遠隔操作プログラム
from quest_dobot_teleop import clamp  # 値を範囲内に収める関数


def quat_mul(a, b):
    """クォータニオンの積 a*b（各要素は [x, y, z, w]）"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def twist_deg(q0, q):
    """
    q0（握った瞬間の向き）から q（今の向き）までの回転のうち、
    コントローラの「指している方向の軸」（targetRaySpace の -Z）まわりの成分 [deg] を返す
    """
    q0_inv = [-q0[0], -q0[1], -q0[2], q0[3]]
    r = quat_mul(q0_inv, q)                 # コントローラから見た相対回転
    ang = 2.0 * math.degrees(math.atan2(-r[2], r[3]))   # -Z 軸まわりの回転角
    return lc.wrap_deg(ang)


class RobotTeleopRx(qt.RobotTeleop):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.q0 = None            # 握った瞬間のコントローラの向き
        self.ref = None           # Rx の基準姿勢 [Rx,Ry,Rz]（setup で決める）
        self.rx0 = 0.0            # 握った瞬間の Rx（基準姿勢から X 軸まわりに回った角度 [deg]）
        self.cmd_rx = 0.0         # 現在の Rx 指令（同じく基準姿勢からの角度 [deg]）
        self.warned_old_page = False   # 「古いページ」の注意を表示したか
        self.last_rx_print = 0.0       # Rx を最後に表示したときの値（表示しすぎ防止）
        # ツール（A=DO1 / B=吸気）の設定
        self.tool = lc.TOOLS[cfg["name"]]                    # name, mode
        self.tool_on, self.tool_off = lc.tool_commands(cfg)  # ON / OFF のコマンド
        # self.suction は手順書の変数名のまま「ツールが ON か」を表す（記録にも使う）

    def set_suction(self, on):
        """ツールを ON / OFF する（手順書の set_suction を、ロボットごとのコマンドに置き換え）"""
        self.bot.send(self.tool_on if on else self.tool_off)
        self.suction = on
        print(f"[{self.name}] {self.tool['name']} {'ON' if on else 'OFF'}")

    def setup(self):
        """接続・イネーブル（手順書と同じ）のあと、ツールを初期状態にする"""
        super().setup()                                      # この中でツールは一度 OFF になる
        if lc.tool_default(self.name):                       # hold_off のツールは ON から始める
            self.set_suction(True)
        # Rx の基準姿勢を決める（ROT_REF が None なら今の姿勢）
        pose = self.bot.get_pose()
        self.ref = lc.rot_ref(self.name, pose)
        # 今の姿勢が基準から RX_RANGE 以上回っていたら、ROT_REF の設定が合っていない
        rx_now = lc.rx_angle(pose, self.ref)
        if abs(rx_now) > lc.RX_RANGE:
            print(f"[{self.name}] 注意: 今の姿勢は基準（ROT_REF）から X 軸まわりに {rx_now:+.1f}° 回っています。"
                  f"初期位置で get_pose.py を実行し、ROT_REF を設定し直してください")

    def update_tool(self, h):
        """トリガーの状態からツールを操作する"""
        trig = h is not None and h.get("trigger", 0.0) >= config.TRIGGER_ON   # 引いているか
        if self.tool["mode"] == "hold_off":
            # 引いている間だけ OFF、離すと ON（通信が途絶えたときも ON に戻す）
            want = not trig
            if want != self.suction:                         # 変わったときだけ送る
                self.set_suction(want)
        elif trig and not self.prev_trig:
            # toggle：引いた瞬間に ON / OFF を切り替え（手順書と同じ）
            self.set_suction(not self.suction)
        self.prev_trig = trig

    def step(self, dt):
        h = qt.get_hand(self.hand)
        self.update_tool(h)                   # トリガー → ツール（DO1・吸気）
        if h is None:                         # 通信が途絶えた → 停止
            self.engaged = False
            return

        # グリップ：握っている間だけ追従
        if not h.get("grip", False):
            self.engaged = False
            return

        pos, rot = h["pos"], h.get("rot")     # 位置 [m] と向き（クォータニオン）
        if h.get("ver", 1) < 2 and not self.warned_old_page:   # 古い index.html のままでは Rx が動かない
            print(f"[{self.name}] 注意: Quest のページが古い版です。Quest Browser でページを再読み込みしてください")
            self.warned_old_page = True
        if not self.engaged:                  # 握った瞬間に基準を記録
            self.p0 = pos                     # 握った瞬間のコントローラ位置
            self.q0 = rot                     # 握った瞬間のコントローラの向き
            self.r0 = self.bot.get_pose()     # 握った瞬間のロボットの姿勢
            self.cmd = list(self.r0)          # 位置の指令はここから始める
            self.rx0 = lc.rx_angle(self.r0, self.ref)   # 今の Rx（基準姿勢からの角度）
            self.cmd_rx = self.rx0            # Rx の指令もここから始める
            self.engaged = True
            return

        # ---- 位置 X, Y, Z（手順書と同じ計算） ----
        k = 1000.0 * config.SCALE             # m → mm（倍率をかける）
        s = self.cfg["axis_sign"]             # 軸の向き（手順書 config.py）
        d = [
            -(pos[2] - self.p0[2]) * k * s["x"],   # 前に押す → X+
            -(pos[0] - self.p0[0]) * k * s["y"],   # 左へ動かす → Y+
            (pos[1] - self.p0[1]) * k * s["z"],    # 上げる → Z+
        ]
        lim = self.cfg["limits"]              # 動作範囲 [mm]
        goal = [                              # 目標位置（範囲内に収める）
            clamp(self.r0[0] + d[0], *lim["x"]),
            clamp(self.r0[1] + d[1], *lim["y"]),
            clamp(self.r0[2] + d[2], *lim["z"]),
        ]
        max_step = config.MAX_SPEED * dt      # 1周期で動ける距離 [mm]
        for i in range(3):
            self.cmd[i] += clamp(goal[i] - self.cmd[i], -max_step, max_step)   # 速度制限つきで近づける

        # ---- Rx（手首ひねり） ----
        if rot is not None and self.q0 is not None:
            tw = twist_deg(self.q0, rot) * lc.RX_SIGN[self.name] * lc.RX_SCALE   # ひねった角度 [deg]
            goal_rx = clamp(self.rx0 + tw, -lc.RX_RANGE, lc.RX_RANGE)            # 目標 Rx（範囲内）
            max_rx = lc.RX_MAX_SPEED * dt                                        # 1周期で回せる角度
            self.cmd_rx += clamp(goal_rx - self.cmd_rx, -max_rx, max_rx)         # 速度制限つきで近づける
            if abs(self.cmd_rx - self.last_rx_print) >= 5.0:                   # 5° 変わるごとに表示
                print(f"[{self.name}] Rx {self.cmd_rx:+.0f}°（手首のひねり {tw:+.0f}°）")
                self.last_rx_print = self.cmd_rx

        # 握った瞬間の姿勢を、ベースの X 軸まわりに (cmd_rx - rx0) 度だけ回した姿勢を送る
        rot = lc.rotate_x(self.r0[3:6], self.cmd_rx - self.rx0)   # [Rx, Ry, Rz]
        pose = self.cmd[:3] + rot
        if config.USE_SERVO:
            self.bot.servo_p(pose)            # ServoP で滑らかに追従
        elif self.bot.robot_mode() == 5:      # MovL 方式は停止中のときだけ次を送る
            self.bot.movl(pose)
```

### 4-7. `record_dataset.py`（遠隔操作しながらデータ収集）

```python
# =====================================================================
#  record_dataset.py
#  Quest で2台の E6 を遠隔操作しながら、LeRobot 形式のデータセットを記録する
#
#  【しくみ】
#   ・遠隔操作は teleop_rx.py（手順書の遠隔操作＋手首ひねりで Rx）を使う
#   ・それと並行して、1秒に FPS 回（10回）次の内容を記録する
#       観測 observation.state  : 2台の現在位置 X,Y,Z [mm]・Rx のずれ [deg]・ツール状態 [0/1]
#       観測 observation.images : カメラ画像（top, front）
#       行動 action             : 2台の目標 X,Y,Z [mm]・目標 Rx のずれ [deg]・ツール指令 [0/1]
#       ツール：ロボットA = 吸気（引くたびに切替）、ロボットB = DO1（トリガーを引いている間だけ OFF）
#
#  【操作】（Quest のボタンと PC のキーボードのどちらでも操作できる）
#    右コントローラ A ／ キー r : 記録開始
#    右コントローラ A ／ キー n : 記録終了して保存（記録中に押す）
#    右コントローラ B ／ キー x : 記録を破棄（やり直し）
#    左コントローラ X ／ キー h : 2台を初期位置へ（記録中は無効）
#    キー c : エラー解除   キー s : 吸気 OFF（ロボットA）   キー q : 終了
#
#  使い方： python record_dataset.py
# =====================================================================

import asyncio                        # Web サーバ（非同期処理）
import json                           # データセットの情報ファイル（info.json）を読む
import os                             # ファイルの場所
import shutil                         # 古いデータセットのフォルダを移す
import ssl                            # HTTPS（Quest の WebXR は HTTPS が必須）
import threading                      # スレッド（並行して動く処理）
import time                           # 周期の調整・経過時間

import numpy as np                    # 観測・行動を数値の配列にする
from aiohttp import web               # Web サーバ

import config                         # 手順書の設定
import lerobot_config as lc           # 模倣学習の設定
import quest_dobot_teleop as qt       # 手順書の遠隔操作プログラム（部品として利用）
from teleop_rx import RobotTeleopRx   # Rx（手首ひねり）対応の遠隔操作
from camera import open_cameras       # USB カメラを開く
from e6_common import KeyReader, dataset_features, power_on_if_needed   # 共通部品
from lerobot.datasets import LeRobotDataset   # LeRobot のデータセット


# ---------------------------------------------------------------------
#  データセットを開く（初回は新規作成、2回目以降は続きから追加）
# ---------------------------------------------------------------------
def open_dataset():
    info = os.path.join(lc.DATASET_ROOT, "meta", "info.json")   # データセットの情報ファイル
    if os.path.exists(info):
        # 既存のデータセットの記録項目が、今の設定（STATE_NAMES）と同じか確認する
        with open(info, encoding="utf-8") as f:
            meta = json.load(f)
        names = meta["features"]["observation.state"].get("names")
        if names != lc.STATE_NAMES:
            if meta.get("total_episodes", 0) == 0:
                # エピソードが1つもない（途中で止まった）→ 別名で残して新しく作り直す
                old = lc.DATASET_ROOT + "_old_" + time.strftime("%Y%m%d_%H%M%S")
                shutil.move(lc.DATASET_ROOT, old)
                print(f"記録項目が変わったため、空のデータセットを {old} に移しました")
            else:
                # エピソードがある → 混ぜると学習できないので止める
                print("既存のデータセットと記録項目が違います")
                print(f"  既存: {names}")
                print(f"  設定: {lc.STATE_NAMES}")
                print(f"  {lc.DATASET_ROOT} を別の場所へ移すか、lerobot_config.py の DATASET_ROOT を変えてください")
                raise SystemExit(1)
    if os.path.exists(info):                                    # すでにある → 続きから追加
        ds = LeRobotDataset.resume(lc.DATASET_REPO_ID, root=lc.DATASET_ROOT,
                                   image_writer_threads=4)
        print(f"既存のデータセットに追加します（現在 {ds.num_episodes} エピソード）")
    else:                                                       # ない → 新規作成
        ds = LeRobotDataset.create(
            repo_id=lc.DATASET_REPO_ID,
            fps=lc.FPS,
            features=dataset_features(),
            root=lc.DATASET_ROOT,
            robot_type=lc.ROBOT_TYPE,
            use_videos=True,             # 画像は MP4 動画として保存（容量が小さい）
            image_writer_threads=4,      # 画像の書き込みを別スレッドで行い、記録周期を守る
        )
        print(f"データセットを新規作成しました: {lc.DATASET_ROOT}")
    return ds


# ---------------------------------------------------------------------
#  Web サーバ（Quest の操作ページと WebSocket）を別スレッドで起動
# ---------------------------------------------------------------------
async def index_nocache(request):
    """操作ページを返す（Quest Browser が古いページを使い回さないよう、キャッシュを禁止する）"""
    resp = web.FileResponse(os.path.join(qt.BASE, "static", "index.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


def start_web_server():
    loop = asyncio.new_event_loop()                   # Web サーバ専用の処理ループ

    async def serve():
        app = web.Application()
        app.router.add_get("/", index_nocache)        # https://PC:8443/   → 操作ページ（index.html）
        app.router.add_get("/ws", qt.ws_handler)      # wss://PC:8443/ws   → コントローラ情報の受信
        sslctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        sslctx.load_cert_chain(os.path.join(qt.BASE, "cert.pem"), os.path.join(qt.BASE, "key.pem"))  # 手順書で作った証明書
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", config.PORT, ssl_context=sslctx).start()

    loop.run_until_complete(serve())                  # サーバを起動
    threading.Thread(target=loop.run_forever, daemon=True).start()   # 以後は別スレッドで動かし続ける
    print(f"Quest のブラウザで https://<PCのWi-Fi IP>:{config.PORT}/ を開いてください")


# ---------------------------------------------------------------------
#  1フレーム分の観測と行動を作る
# ---------------------------------------------------------------------
def make_frame(teleops, cams):
    state, action = [], []
    for t in teleops:
        pose = t.bot.get_pose()                       # 現在の姿勢 [X,Y,Z,Rx,Ry,Rz]
        rx = lc.rx_angle(pose, t.ref)                 # Rx は基準姿勢から X 軸まわりに回った角度 [deg]
        suction = 1.0 if t.suction else 0.0           # ツール（A=DO1 / B=吸気）が ON なら 1
        state += [pose[0], pose[1], pose[2], rx, suction]
        cmd = t.cmd
        if t.engaged and cmd is not None:             # グリップ中：遠隔操作の指令が「行動」
            target = [cmd[0], cmd[1], cmd[2], t.cmd_rx]
        else:                                         # グリップを離している：その場で停止＝現在の姿勢
            target = [pose[0], pose[1], pose[2], rx]
        action += target + [suction]

    frame = {
        "observation.state": np.array(state, dtype=np.float32),   # 10個の数値
        "action": np.array(action, dtype=np.float32),             # 10個の数値
        "task": lc.TASK,                                          # 作業の説明文
    }
    for name, cam in cams.items():
        frame[f"observation.images.{name}"] = cam.read()          # カメラ画像（RGB）
    return frame


# ---------------------------------------------------------------------
#  Quest のボタンが「押された瞬間」を調べる
# ---------------------------------------------------------------------
class ButtonEdges:
    def __init__(self):
        self.prev = {}                                # 前回の押下状態（手・ボタンごと）

    def pressed(self, hand, button):
        """今回押されていて、前回は押されていなければ True（押した瞬間だけ反応）"""
        h = qt.get_hand(hand)                         # Quest から届いた最新のコントローラ情報
        now = bool(h and h.get(button, False))
        key = (hand, button)
        was = self.prev.get(key, False)
        self.prev[key] = now
        return now and not was


def main():
    # ---- 準備 ----
    ds = open_dataset()                               # データセットを開く（新規 or 続き）
    cams = open_cameras(lc.CAMERAS)                   # カメラ2台を開く

    for cfg in config.ROBOTS:
        power_on_if_needed(cfg)                       # 電源 OFF なら投入して待つ
    # 遠隔操作スレッド（手順書の RobotTeleop に Rx 制御を加えた RobotTeleopRx）
    teleops = [RobotTeleopRx(cfg) for cfg in config.ROBOTS]
    for t in teleops:
        t.setup()                                     # 接続・イネーブル・ツールを初期状態に
    for t in teleops:
        t.start()                                     # 20Hz の遠隔操作ループを開始
    start_web_server()                                # Quest から接続できるようにする

    keys = KeyReader()                                # キーボード入力の受付を開始
    keys.start()
    buttons = ButtonEdges()                           # Quest のボタンの「押した瞬間」を調べる

    print("操作: 右A/r=記録開始  右A/n=保存  右B/x=破棄  左X/h=初期位置  c=エラー解除  s=吸気OFF(A)  q=終了")
    print(f"現在 {ds.num_episodes} エピソード記録済み。初期位置（h）に戻してから記録を始めてください")

    recording = False                                 # 記録中かどうか
    n_frames = 0                                      # このエピソードで記録したフレーム数
    t_start = 0.0                                     # 記録を始めた時刻
    period = 1.0 / lc.FPS                             # 1周期の時間（0.1秒）
    late = 0                                          # 周期に間に合わなかった回数

    def save():
        """記録中のエピソードを保存する（動画の作成に数秒かかる）"""
        nonlocal recording
        recording = False
        print(f"\n保存中…（{n_frames} フレーム、動画を作成しています）")
        ds.save_episode()
        print(f"保存しました → 合計 {ds.num_episodes} エピソード")

    def discard():
        """記録中のエピソードを捨てる（失敗したとき）"""
        nonlocal recording
        recording = False
        ds.clear_episode_buffer()
        print("\n記録を破棄しました。h で初期位置に戻し、やり直してください")

    try:
        while True:
            t0 = time.time()                          # この周期の開始時刻

            # ---- 入力の処理（キーボードと Quest のボタン） ----
            k = keys.get()                            # 押されたキー（なければ None）
            right_a = buttons.pressed("right", "a")   # 右コントローラ A
            right_b = buttons.pressed("right", "b")   # 右コントローラ B
            left_x = buttons.pressed("left", "a")     # 左コントローラの X は "a" として届く

            if k == "q":                              # 終了（記録中なら破棄してから）
                if recording:
                    discard()
                break
            elif (k == "r" or right_a) and not recording:      # 記録開始
                recording, n_frames, t_start, late = True, 0, time.time(), 0
                print(f"● 記録開始（エピソード {ds.num_episodes}）")
                right_a = False                       # 同じ押下で終了しないように
            if recording and (k == "n" or right_a):            # 記録終了・保存
                save()
            elif recording and (k == "x" or right_b):          # 記録の破棄
                discard()
            elif (k == "h" or left_x) and not recording:       # 初期位置へ（記録中は無効）
                for t in teleops:
                    t.home_req.set()
            elif k == "c":                                     # エラー解除・再イネーブル
                for t in teleops:
                    t.engaged = False
                    t.bot.enable()
            elif k == "s":                                     # 吸気 OFF（切替式のツールだけ）
                for t in teleops:
                    if t.tool["mode"] == "toggle":             # hold_off のツールはトリガーで決まるので対象外
                        t.set_suction(False)

            # ---- 記録（1周期に1フレーム） ----
            if recording:
                try:
                    ds.add_frame(make_frame(teleops, cams))    # 観測・行動・画像を追加
                    n_frames += 1
                    if n_frames % lc.FPS == 0:                 # 1秒ごとに経過時間を表示
                        print(f"  記録中 {n_frames // lc.FPS} 秒", end="\r")
                except Exception as e:                         # カメラ・通信の異常
                    print(f"記録エラー: {e} → このエピソードは破棄します")
                    discard()
                if recording and time.time() - t_start > lc.EPISODE_TIME_SEC:
                    print("\n制限時間に達しました")
                    save()                                     # 制限時間で自動保存

            # ---- 周期の調整（1周期 0.1 秒になるよう待つ） ----
            spent = time.time() - t0
            if recording and spent > period:                   # 処理が間に合わなかった
                late += 1
                if late % 20 == 1:                             # 表示しすぎないよう20回に1回
                    print(f"注意: 記録周期に遅れています（{spent * 1000:.0f} ms）")
            time.sleep(max(0.0, period - spent))
    finally:
        # ---- 終了処理（エラーで止まった場合も必ず実行） ----
        print("終了処理中…")
        ds.finalize()                                 # データセットのファイルを正しく閉じる（必須）
        for t in teleops:
            t.running = False                         # 遠隔操作ループを止める
        for t in teleops:
            t.join(timeout=2)                         # スレッドの終了を待つ
            try:
                t.set_suction(False)                  # ツール OFF（DO1・吸気とも）
            except Exception:
                pass
            t.bot.close()                             # 切断
        for c in cams.values():
            c.close()                                 # カメラを閉じる
        print(f"終了しました（合計 {ds.num_episodes} エピソード）")


if __name__ == "__main__":
    main()
```

### 4-8. `check_dataset.py`（データセットの確認）

```python
# =====================================================================
#  check_dataset.py
#  記録したデータセットの中身を確認する
#
#  ・エピソード数、フレーム数、各エピソードの長さを表示
#  ・指定エピソードの「位置・Rx・ツール（A=DO1, B=吸気）の変化」をグラフにして保存
#  ・指定エピソードの最初・中間・最後のカメラ画像を保存
#
#  使い方： python check_dataset.py          （エピソード 0 を確認）
#          python check_dataset.py 5        （エピソード 5 を確認）
#  結果は dataset_check フォルダに保存されます
# =====================================================================

import os                                          # フォルダの作成
import sys                                         # コマンドの引数（エピソード番号）

import matplotlib
matplotlib.use("Agg")                              # 画面を使わずに画像ファイルへ描く
import matplotlib.pyplot as plt                    # グラフを描く
import numpy as np                                 # 数値の配列
from PIL import Image                              # 画像の保存

import lerobot_config as lc                        # 模倣学習の設定
from lerobot.datasets import LeRobotDataset        # LeRobot のデータセット

ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 確認するエピソード番号（指定がなければ 0）
OUT = os.path.join(lc.BASE, "dataset_check")       # 結果を保存するフォルダ
os.makedirs(OUT, exist_ok=True)

ds = LeRobotDataset(lc.DATASET_REPO_ID, root=lc.DATASET_ROOT)   # 記録したデータセットを開く

# ---- 全体の概要 ----
print(f"データセット : {lc.DATASET_ROOT}")
print(f"エピソード数 : {ds.num_episodes}")
print(f"総フレーム数 : {ds.num_frames}（{ds.num_frames / ds.fps / 60:.1f} 分）")
print(f"fps          : {ds.fps}")
print("記録項目     :", ", ".join(ds.features.keys()))

# 数値データ（画像以外）をまとめて取り出す
cols = ds.hf_dataset.with_format("numpy")        # 数値の列を numpy で取り出す
ep_idx = np.asarray(cols["episode_index"])       # 各フレームのエピソード番号
state = np.stack(cols["observation.state"])     # 全フレームの状態（フレーム数 × 10）
action = np.stack(cols["action"])               # 全フレームの行動（フレーム数 × 10）

lengths = [int((ep_idx == e).sum()) for e in range(ds.num_episodes)]   # エピソードごとのフレーム数
print("各エピソードの長さ [秒]:")
print("  " + "  ".join(f"{i}:{n / ds.fps:.0f}" for i, n in enumerate(lengths)))
short = [i for i, n in enumerate(lengths) if n < 5 * ds.fps]   # 5秒未満のエピソード
if short:
    print(f"注意: 5秒未満のエピソードがあります → {short}（操作ミスの可能性。確認してください）")

# ---- 各値の範囲（動作範囲の外れ値がないか確認） ----
print("各値の範囲（最小 〜 最大）:")
for i, name in enumerate(lc.STATE_NAMES):
    print(f"  {name:10s} 状態 {state[:, i].min():8.1f} 〜 {state[:, i].max():8.1f}   "
          f"行動 {action[:, i].min():8.1f} 〜 {action[:, i].max():8.1f}")

# ---- 指定エピソードのグラフ ----
sel = ep_idx == ep                                 # 指定エピソードのフレームだけ True
if not sel.any():
    sys.exit(f"エピソード {ep} はありません")
t = np.arange(sel.sum()) / ds.fps                  # 横軸：時間 [秒]
n = lc.N_PER_ROBOT                                 # 1台あたりの数（5）→ 5行のグラフ
fig, axes = plt.subplots(n, 2, figsize=(11, 2.2 * n), sharex=True)
for i, name in enumerate(lc.STATE_NAMES):
    ax = axes[i % n, i // n]                       # 左列＝ロボットA、右列＝ロボットB
    ax.plot(t, state[sel, i], label="state (observed)", color="tab:blue")      # 実際の値（青）
    ax.plot(t, action[sel, i], label="action (target)", color="tab:orange", linestyle="--")   # 目標（橙の破線）
    ax.set_title(name)                             # グラフの題名（A.x など）
    ax.grid(alpha=0.3)
axes[0, 0].legend(loc="best", fontsize=8)          # 凡例は左上のグラフだけに表示
for ax in axes[-1]:
    ax.set_xlabel("time [s]")
fig.suptitle(f"Episode {ep}")
fig.tight_layout()
png = os.path.join(OUT, f"episode_{ep:03d}_plot.png")
fig.savefig(png, dpi=120, facecolor="white")      # 印刷用に白背景
print(f"グラフを保存: {png}")

# ---- 指定エピソードのカメラ画像（最初・中間・最後） ----
first = int(np.argmax(sel))                        # 指定エピソードの最初のフレーム番号
idxs = {"start": first, "middle": first + lengths[ep] // 2, "end": first + lengths[ep] - 1}
for when, idx in idxs.items():
    item = ds[idx]                                 # 1フレーム分（動画から画像を取り出す）
    for key in ds.meta.camera_keys:
        img = item[key]                            # (3, 高さ, 幅) の 0〜1 の値
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # (高さ, 幅, 3) の 0〜255 に戻す
        name = key.split(".")[-1]                  # "top" または "front"
        path = os.path.join(OUT, f"episode_{ep:03d}_{when}_{name}.jpg")
        Image.fromarray(arr).save(path)
print(f"画像を保存: {OUT}")
```

### 4-9. `replay_episode.py`（記録した動きの再生）

```python
# =====================================================================
#  replay_episode.py
#  記録したエピソードの「行動（action）」を、そのまま2台のロボットで再生する
#
#  ・データが正しく記録できているかの確認に使う
#    （再生して同じ作業ができれば、AI に渡すデータとして正しい）
#  ・ワークと箱は記録したときと同じ位置に置き、蓋も同じ角度に開けてから再生する
#  ・速度は lerobot_config.AUTO_MAX_SPEED / AUTO_RX_MAX_SPEED で制限される
#
#  使い方： python replay_episode.py 3     （エピソード 3 を再生）
#  キー操作： Enter=開始  q=中断（ツール OFF・停止）
# =====================================================================

import sys                                           # コマンドの引数（エピソード番号）を読む
import time                                          # 再生の周期を調整する

import numpy as np                                   # 行動データを配列で扱う

import config                                        # 手順書の設定
import lerobot_config as lc                          # 模倣学習の設定
from e6_common import ArmDriver, KeyReader, power_on_if_needed   # 共通部品
from lerobot.datasets import LeRobotDataset          # データセットの読み込み

# 再生するエピソード番号（指定がなければ 0）
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ---- 行動データの読み込み（画像は使わない） ----
ds = LeRobotDataset(lc.DATASET_REPO_ID, root=lc.DATASET_ROOT)
cols = ds.hf_dataset.with_format("numpy")            # 数値の列だけを numpy で取り出す
ep_idx = np.asarray(cols["episode_index"])           # 各フレームが何番のエピソードか
actions = np.stack(cols["action"])[ep_idx == ep]     # 指定エピソードの行動だけを取り出す
if len(actions) == 0:
    sys.exit(f"エピソード {ep} はありません")
print(f"エピソード {ep}: {len(actions)} フレーム（{len(actions) / ds.fps:.1f} 秒）")

# ---- ロボットの準備 ----
for cfg in config.ROBOTS:
    power_on_if_needed(cfg)                          # 電源 OFF なら投入して待つ
drivers = [ArmDriver(cfg, lc.AUTO_MAX_SPEED, lc.AUTO_RX_MAX_SPEED) for cfg in config.ROBOTS]
for d in drivers:
    d.setup()                                        # 接続・イネーブル・ツールを初期状態に
print("2台を初期位置へ移動します")
for d in drivers:
    d.go_home()                                      # 記録と同じく初期位置から始める

keys = KeyReader()                                   # キー入力（Enter・q）を受け付ける
keys.start()
print("ワークと箱を記録時と同じ位置に置き、Enter を押すと再生を開始します（中断は q）")
if not keys.wait_enter():                            # Enter を待つ（q なら再生しないで終わる）
    print("再生をやめました")
    for d in drivers:
        d.shutdown()                                 # 停止・ツール OFF・切断
    sys.exit(0)
print("再生を開始します")

try:
    for d in drivers:
        d.begin()                                    # 現在の姿勢から追従開始
    period = 1.0 / ds.fps                            # 記録と同じ周期（0.1秒）で再生
    n = lc.N_PER_ROBOT                               # 1台あたりの数（5）
    for i, a in enumerate(actions):
        t0 = time.time()
        if keys.get() == "q":                        # 中断
            print("中断しました")
            break
        # a = [A.x, A.y, A.z, A.rx, A.do1, B.x, B.y, B.z, B.rx, B.suction]
        for j, d in enumerate(drivers):
            x, y, z, rx, s = a[n * j: n * j + n]     # j 台目の5つの値
            d.set_target([x, y, z], rx, s >= lc.SUCTION_THRESHOLD)
        time.sleep(max(0.0, period - (time.time() - t0)))
    else:
        time.sleep(1.0)                              # 最後の目標に追いつくまで少し待つ
        print("再生が終わりました")
finally:
    for d in drivers:
        d.shutdown()                                 # 停止・ツール OFF・切断
```

### 4-10. `run_policy.py`（学習済みモデルで自律動作）

```python
# =====================================================================
#  run_policy.py
#  学習済みモデル（ACT）で2台の E6 を自律動作させる
#
#  【しくみ】 1秒に FPS 回（10回）次をくり返す
#   1. カメラ画像2枚と、2台の現在位置・Rx・ツール状態（A=DO1, B=吸気）を読む（観測）
#   2. 学習済みモデルに観測を入れ、次の行動（目標位置・目標 Rx・ツール ON/OFF）を受け取る
#   3. ArmDriver に目標を渡す（動作範囲・速度・Rx の範囲を制限して ServoP で移動）
#
#  【キー操作】（このコマンドプロンプトがアクティブなときのみ）
#    スペース : 自律動作 開始 ／ 一時停止
#    h : 2台を初期位置へ（一時停止してから移動）
#    s : ツール OFF（DO1・吸気）   c : エラー解除   q : 終了
#
#  【安全】補助者は必ず非常停止スイッチの前で待機すること
#  使い方： python run_policy.py
# =====================================================================

import time                                          # 周期の調整・経過時間の計測

import numpy as np                                   # 観測を数値の配列にする
import torch                                         # 学習済みモデルを動かすライブラリ

import config                                        # 手順書の設定（IP・動作範囲・吸気コマンド）
import lerobot_config as lc                          # 模倣学習の設定
from camera import open_cameras                      # USB カメラを開く
from e6_common import ArmDriver, KeyReader, power_on_if_needed   # 共通部品
from lerobot.common.control_utils import predict_action          # 観測 → 行動 を1回計算する関数
from lerobot.policies.act.modeling_act import ACTPolicy          # ACT モデル本体
from lerobot.policies.factory import make_pre_post_processors    # 正規化などの前処理・後処理


def load_policy():
    """学習済みモデルと、その前処理・後処理（正規化など）を読み込む"""
    print(f"モデルを読み込みます: {lc.POLICY_PATH}")
    policy = ACTPolicy.from_pretrained(lc.POLICY_PATH)   # pretrained_model フォルダから読み込む
    policy.config.n_action_steps = lc.N_ACTION_STEPS    # 何ステップ実行したら予測し直すか
    policy.to(lc.POLICY_DEVICE)                          # 計算する装置（CPU）へ移す
    policy.eval()                                        # 推論モード（学習用の処理を止める）
    # 前処理：観測を学習時と同じ尺度にそろえる（正規化）
    # 後処理：モデルの出力を mm・deg・0/1 の値に戻す（逆正規化）
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=lc.POLICY_PATH,
        preprocessor_overrides={"device_processor": {"device": lc.POLICY_DEVICE}},
    )
    return policy, pre, post


def read_observation(drivers, cams):
    """記録時と同じ形の観測を作る"""
    state = []
    for d in drivers:                                # ロボットA → B の順（記録時と同じ）
        state += d.read_state()                      # [X, Y, Z, Rx, ツール(A=DO1/B=吸気)]
    obs = {"observation.state": np.array(state, dtype=np.float32)}   # 10個の数値
    for name, cam in cams.items():
        obs[f"observation.images.{name}"] = cam.read()   # (高さ, 幅, 3) の RGB 画像
    return obs


def reset_policy(policy, pre, post):
    """行動の予定（アクションチャンク）を捨てて、最初から予測し直す"""
    policy.reset()
    pre.reset()
    post.reset()


def stop_all(drivers):
    """2台とも追従を止める（その場で停止）"""
    for d in drivers:
        d.pause()


def main():
    # ---- 準備 ----
    policy, pre, post = load_policy()                # モデルの読み込み
    device = torch.device(lc.POLICY_DEVICE)
    cams = open_cameras(lc.CAMERAS)                  # カメラ2台を開く
    for cfg in config.ROBOTS:
        power_on_if_needed(cfg)                      # 電源 OFF なら投入して待つ
    # 自律動作用の速度制限（AUTO_MAX_SPEED, AUTO_RX_MAX_SPEED）で ArmDriver を作る
    drivers = [ArmDriver(cfg, lc.AUTO_MAX_SPEED, lc.AUTO_RX_MAX_SPEED) for cfg in config.ROBOTS]
    for d in drivers:
        d.setup()                                    # 接続・イネーブル・ツールを初期状態に

    keys = KeyReader()                               # キーボード入力の受付を開始
    keys.start()
    print("キー操作: スペース=開始/一時停止  h=初期位置  s=ツールOFF  c=エラー解除  q=終了")
    print("まず h で初期位置へ移動し、ワークと箱を置いてからスペースで開始してください")

    running = False                                  # 自律動作中かどうか
    t_start = 0.0                                    # 自律動作を始めた時刻
    period = 1.0 / lc.FPS                            # 1周期の時間（0.1秒）
    try:
        while True:
            t0 = time.time()                         # この周期の開始時刻

            # ---- キー入力の処理 ----
            k = keys.get()
            if k == "q":                             # 終了
                break
            elif k == " ":                           # 開始 ／ 一時停止 の切り替え
                if running:
                    running = False
                    stop_all(drivers)
                    print("一時停止")
                else:
                    reset_policy(policy, pre, post)  # 前回の予定を捨てる
                    for d in drivers:
                        d.begin()                    # 現在の姿勢から追従開始
                    running, t_start = True, time.time()
                    print("▶ 自律動作 開始")
            elif k == "h":                           # 初期位置へ
                running = False
                print("初期位置へ移動します")
                for d in drivers:
                    d.go_home()                      # 移動が終わるまで待つ
                print("初期位置に到着")
            elif k == "s":                           # ツール OFF（DO1・吸気。一時停止もする）
                running = False
                for d in drivers:
                    d.pause()
                    d._set_suction(False)
                print("ツール OFF（一時停止しました）")
            elif k == "c":                           # エラー解除・再イネーブル
                running = False
                for d in drivers:
                    d.pause()
                    d.bot.enable()
                print("エラー解除しました")

            # ---- 自律動作（観測 → 推論 → 目標を渡す） ----
            if running:
                obs = read_observation(drivers, cams)
                t_inf = time.time()                  # 推論にかかった時間を測る
                action = predict_action(obs, policy, device, pre, post, use_amp=False,
                                        task=lc.TASK, robot_type=lc.ROBOT_TYPE)
                a = action.squeeze(0).cpu().numpy()  # [A.x, A.y, A.z, A.rx, A.do1, B.x, …, B.suction]
                n = lc.N_PER_ROBOT                   # 1台あたりの数（5）
                for j, d in enumerate(drivers):
                    x, y, z, rx, s = a[n * j: n * j + n]          # j 台目の5つの値を取り出す
                    d.set_target([x, y, z], rx, s >= lc.SUCTION_THRESHOLD)
                dt_inf = time.time() - t_inf
                if dt_inf > period:                  # 予測し直すときは時間がかかる
                    print(f"  推論 {dt_inf * 1000:.0f} ms（予測し直しのタイミング）")
                if time.time() - t_start > lc.RUN_TIME_SEC:       # 制限時間を超えたら止める
                    running = False
                    stop_all(drivers)
                    print("制限時間に達したので停止しました")

            # ---- 周期の調整（1周期 0.1 秒になるよう待つ） ----
            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        # ---- 終了処理（エラーで止まった場合も必ず実行） ----
        for d in drivers:
            d.shutdown()                             # 停止・ツール OFF・切断
        for c in cams.values():
            c.close()                                # カメラを閉じる
        print("終了しました")


if __name__ == "__main__":
    main()
```

### 4-11. `static\index.html`（差し替え）

手順書の `index.html` に、コントローラの向き（Rx 用）と A/X・B/Y ボタンの送信、操作表を追加したものです。遠隔操作（`quest_dobot_teleop.py`）でもそのまま使えます。

```html
<!doctype html>
<!--
  static/index.html（模倣学習対応版）
  手順書の index.html に「コントローラの向き」「A/X ボタン」「B/Y ボタン」の送信を追加したものです。
  遠隔操作（quest_dobot_teleop.py）でもそのまま使えます（追加の値は無視されます）。

  送信する内容（約30回/秒）:
    hands.left / hands.right = {
      pos: [x, y, z]  コントローラの位置 [m]
      rot: [x, y, z, w] コントローラが指している方向の向き（クォータニオン。Rx の手首ひねりに使う）
      ver: 2          このページの版（古いページのままか、PC 側で確認するため）
      trigger: 0〜1   人差し指トリガーの引き量
      grip: true/false 中指グリップを握っているか
      a: true/false   右=A ボタン / 左=X ボタン
      b: true/false   右=B ボタン / 左=Y ボタン
    }
-->
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quest DOBOT Teleop</title>
<style>
  body { font-family: sans-serif; background: #fff; color: #222; padding: 24px; }
  button { font-size: 28px; padding: 16px 32px; }
  #status { margin-top: 16px; font-size: 20px; }
  table { margin-top: 16px; border-collapse: collapse; font-size: 18px; }
  td { border: 1px solid #999; padding: 4px 12px; }
</style>
</head>
<body>
<h1>DOBOT E6 ×2 遠隔操作（データ収集・Rx 対応）</h1>
<p>ロボットの正面と同じ方向を向いてから「開始」を押してください。</p>
<button id="enter">開始</button>
<div id="status">サーバ接続中…</div>
<table>
  <tr><td>右 A</td><td>記録開始 ／ 記録終了・保存</td></tr>
  <tr><td>右 B</td><td>記録を破棄</td></tr>
  <tr><td>左 X</td><td>2台を初期位置へ</td></tr>
  <tr><td>手首をひねる</td><td>Rx（ベースの X 軸まわりにノズルを傾ける）</td></tr>
  <tr><td>左トリガー</td><td>ロボットA の DO1（引くたびに ON / OFF）</td></tr>
  <tr><td>右トリガー</td><td>ロボットB の吸気（引くたびに ON / OFF）</td></tr>
</table>

<script>
const statusEl = document.getElementById('status');   // 状態表示の欄
let ws = null;                                        // サーバとの通信（WebSocket）

// WebSocket でサーバ（PC）に接続。切れたら1秒後に再接続
function connectWS() {
  ws = new WebSocket(`wss://${location.host}/ws`);
  ws.onopen  = () => statusEl.textContent = 'サーバ接続OK';
  ws.onclose = () => { statusEl.textContent = 'サーバ切断 → 再接続中…'; setTimeout(connectWS, 1000); };
}
connectWS();

// ボタンが押されているか（存在しないボタンは false）
const pressed = (b, i) => !!(b[i] && b[i].pressed);

document.getElementById('enter').onclick = async () => {
  if (!navigator.xr) { alert('WebXR に対応していません'); return; }   // Quest Browser 以外の対策
  // パススルー（immersive-ar）が使えればそちらを優先
  const mode = (await navigator.xr.isSessionSupported('immersive-ar')) ? 'immersive-ar' : 'immersive-vr';
  const session = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor'] });

  // XR の描画先（何も描かないが、WebXR の動作に必要）
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', { xrCompatible: true });
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });

  // 位置の基準（床が原点。使えない場合は開始時の頭の位置）
  let ref;
  try { ref = await session.requestReferenceSpace('local-floor'); }
  catch (e) { ref = await session.requestReferenceSpace('local'); }

  let last = 0;                                         // 前回送信した時刻
  // 毎フレーム（約72〜90回/秒）呼ばれる処理
  session.requestAnimationFrame(function onFrame(t, frame) {
    session.requestAnimationFrame(onFrame);             // 次のフレームも呼ぶ

    // 画面は透明に塗る（パススルーで実機が見える）
    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (t - last < 33) return;   // 約30Hzで送信
    last = t;

    const hands = {};                                   // 左右のコントローラ情報
    for (const src of session.inputSources) {
      if (src.hand || !src.gamepad || !src.gripSpace) continue;   // ハンドトラッキングは使わない
      const pose = frame.getPose(src.gripSpace, ref);  // 握り部分の位置（X,Y,Z の操作に使う）
      if (!pose) continue;                             // 見失っているときは送らない
      const ray = frame.getPose(src.targetRaySpace, ref);   // 指している方向（手首ひねりの軸）
      const p = pose.transform.position;               // 位置 [m]
      // 向き：指している方向の軸まわりの回転＝手首のひねり。grip の向きは持ち手の軸なので使わない
      const q = (ray || pose).transform.orientation;
      const b = src.gamepad.buttons;   // [0]=トリガー [1]=グリップ [4]=A/X [5]=B/Y
      hands[src.handedness] = {                        // "left" または "right"
        pos: [p.x, p.y, p.z],
        rot: [q.x, q.y, q.z, q.w],
        ver: 2,
        trigger: b[0] ? b[0].value : 0,
        grip: b[1] ? b[1].value > 0.5 : false,
        a: pressed(b, 4),
        b: pressed(b, 5)
      };
    }
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ t: Date.now(), hands }));   // サーバへ送信
  });

  session.addEventListener('end', () => statusEl.textContent = 'XR 終了');   // XR を閉じたとき
};
</script>
</body>
</html>
```

### 4-12. `train_act_colab.ipynb`（Colab 学習用ノートブック）

ノートブックは配布フォルダの `train_act_colab.ipynb` を使います。中身と使い方は第8章で説明します。

## 第5章 動作確認

データ収集の前に、カメラ・ロボット・Quest のボタンがすべて動くことを確かめます。ここで問題を取り除いておくと、収集中のやり直しが減ります。

### 5-1. 配布ファイルの配置

1. 配布された `e6_lerobot_imitation.zip` を展開する
2. 中の `.py` ファイル9つと `train_act_colab.ipynb` を `C:\quest_dobot` にコピーする
3. `static\index.html` を、zip 内の `static\index.html` で上書きする（念のため元のファイルは `index_old.html` などに名前を変えて残す）

> 手順書のファイル（`config.py`・`dobot_client.py`・`quest_dobot_teleop.py`・`get_pose.py`・`test_suction.py`・`gen_cert.py`）が `C:\quest_dobot` に無いと、`python get_pose.py` などで「No such file or directory」になります。zip にはこの6つも入っているので、zip の中身をすべて `C:\quest_dobot` にコピーしてください（すでに設定済みの `config.py` がある場合は上書きしない）。このとき `python gen_cert.py <PCのWi-Fi IP>` で証明書も作ります。
>
> zip を展開すると `e6_lerobot_imitation` フォルダができます。その中身を `C:\quest_dobot` 直下に置いてください（フォルダごと置くと動きません）。

#### Quest で操作する前に必要な3つ

Quest で操作する前に、次の3つが必要です。手順書を済ませている場合も、念のため確認してください。

1. **証明書を作る**：Quest の操作ページは HTTPS で開くため、証明書（`cert.pem` / `key.pem`）が必要です。まだ作っていない場合は、PC の Wi-Fi の IP アドレス（`ipconfig` で確認）を指定して作ります。Wi-Fi の IP が変わったら作り直します。

   ```bat
   python gen_cert.py 192.168.1.20
   ```
2. **初期位置（`home_joint`）を設定する**：`config.py` の `home_joint` が `None` のままだと、キー `h` や左X を押しても初期位置に戻りません。DobotStudio Pro などで2台を初期位置にしたい姿勢に動かし、DobotStudio Pro を切断してから `python get_pose.py` を実行します。表示された「関節 J1..J6」の値を、ロボットごとに `config.py` の `home_joint` に書き込みます（手順書 7-1）。

   ```python
   "home_joint": [2.6, -2.9, -118.9, 31.8, 90.0, 2.6],   # ← get_pose.py の値に置き換える
   ```
3. **IP アドレスを確認する**：`config.py` はロボットA＝`192.168.5.1`、ロボットB＝`192.168.5.2` にしてあります。実機の設定と違う場合は `"ip"` を書き換えます。`ping 192.168.5.1` と `ping 192.168.5.2` で応答があることも確認します。

### 5-2. 他のソフトが接続していないことを確認

E6 の制御ポート（29999）は、同時に1つのソフトしか使えません。次のソフトは必ず終了しておきます。

| ソフト | 終了方法 |
| --- | --- |
| DobotStudio Pro | 「切断」してからアプリを終了 |
| ROS 2（Docker）の bringup | `dobot_bringup_ros2.launch.py` のターミナルで Ctrl+C。コンテナごと止めるなら Ubuntu で `docker compose stop` |
| 手順書の `quest_dobot_teleop.py` | キー `q` で終了 |

> 「ROS 2 で動かす手順書」の環境（WSL2 + Docker）とこの教材の環境（Windows の Python）は別物です。この教材では ROS 2 も Docker も使いません。また、PC の有線LAN の IP は手順書どおり `192.168.5.100` にしておきます（ROS 2 版の `192.168.5.10` のままでも同じネットワークなので動作します）。

### 5-3. カメラ番号の確認

```bat
cd C:\quest_dobot
.venv\Scripts\activate
python find_cameras.py
```

`camera_check` フォルダの `cam0.jpg`、`cam1.jpg` … を開き、真上カメラと正面カメラの番号を確かめて `lerobot_config.py` の `CAMERAS` に書きます。

```python
CAMERAS = {
    "top":   {"index": 1, "width": 640, "height": 480},   # ← 真上カメラの番号
    "front": {"index": 2, "width": 640, "height": 480},   # ← 正面カメラの番号
}
```

### 5-4. ロボットの確認

手順書のプログラムで、2台の接続とロボットBの吸気を確認します（`test_suction.py` はロボットAの DO1 も ON → OFF と切り替えて確認します）。

```bat
python get_pose.py
python test_suction.py
```

### 5-5. Rx の基準姿勢（ROT\_REF）の設定

手首をひねると、ノズルはロボットのベース座標の X 軸まわりに回ります（ティーチングペンダントで「ベース座標の Rx」をジョグしたときと同じ回り方）。記録する Rx は「基準姿勢から X 軸まわりに何度回したか」です。その基準姿勢（Rx, Ry, Rz）を2台それぞれ設定します。

1. 2台を初期位置（手順書 7-1 で登録した `home_joint`）へ移動させる（手順書の `quest_dobot_teleop.py` を起動してキー `h` を押し、`q` で終了）
2. `python get_pose.py` を実行し、「**ROT\_REF 用」の行**を読む
3. `lerobot_config.py` の `ROT_REF` に書く

```python
ROT_REF = {"A": [179.8, 0.2, 92.6], "B": [-179.6, -0.1, 91.8]}   # ← get_pose.py の「ROT_REF 用」の値（例）
```

> None のままでも動きます（起動したときの姿勢が基準になります）。ただし記録と自律動作で基準がずれないよう、初期位置で読んだ値を書いておきます。姿勢の数値 Rx だけを変えると、ツールの向きによっては Ry 方向に回って見えるため、プログラムは「ベースの X 軸まわりに回した姿勢」を計算して Rx, Ry, Rz をまとめて送ります。そのため、ひねると Ry や Rz の数値も変わることがあります。

### 5-6. Quest のボタン確認（試し記録）

1. 次のコマンドで起動する

   ```bat
   python record_dataset.py
   ```
2. Quest Browser で `https://<PCのWi-Fi IP>:8443/` を開き直す（ページを再読み込みして新しい `index.html` を読み込む）。操作表（右A・右B・左X・手首をひねる・左右トリガー）が表示されれば差し替え成功。起動時に PC に「\[A\] do1 OFF」「\[B\] suction OFF」と表示される
3. 「開始」を押し、グリップで2台が動くことを確認する
4. **Rx の向きの確認**：グリップを握ったまま、手首を右へひねる（後ろから見て時計回り）。ノズルが右へ傾けば正しい。逆に傾く場合は `lerobot_config.py` の `RX_SIGN` のそのロボットを `-1` にする
5. **Rx の範囲の確認**：大きくひねっても、`RX_RANGE`（40°）以上は傾かないことを確認する。傾けすぎて関節が限界に近い場合は `RX_RANGE` を小さくする
6. トリガーの確認：左トリガーを引くたびに PC に「\[A\] do1 ON」「\[A\] do1 OFF」、右トリガーを引くたびに「\[B\] suction ON」「\[B\] suction OFF」と表示され、DobotStudio Pro の IO 画面や接続した機器も同じように切り替わることを確認する（反応しない場合は lerobot\_config.py の TOOLS と config.py の suction\_on / suction\_off を確認）
7. **左X** → 2台が初期位置へ移動する
8. **右A** → PC に「● 記録開始」と表示される
9. 少し動かしてから **右B** → 「記録を破棄しました」と表示される
10. PC でキー `q` を押して終了する

> ノズルを傾けた状態で下げると、ノズルの先が想定より横にずれます。Rx を確認するときは、作業台から 50mm 以上離れた高さで行ってください。

> 起動時に `data\e6_box_lid` フォルダが作られます。試し記録で保存してしまった場合は、`q` で終了したあと、このフォルダを丸ごと削除すれば最初からやり直せます。

### 5-7. 自己チェック

- [ ] `find_cameras.py` で2台のカメラ画像が保存され、番号を設定した
- [ ] `get_pose.py` で2台とも座標が表示され、`RX_CENTER` を設定した
- [ ] Quest のページに操作表（右A・右B・左X・手首をひねる）が表示された
- [ ] 手首を右へひねると、ノズルが右へ傾いた（2台とも）
- [ ] 右A・右B・左X が PC 側で反応した

## 第6章 データ収集

50エピソードを目安に、同じ手順の蓋閉め作業をくり返し記録します。1エピソード30〜50秒、準備を含めて1回1.5〜2分なので、50回で約1.5〜2時間です。

### 6-1. 役割分担

| 役割 | 仕事 |
| --- | --- |
| 操作者 | Quest を装着して2台を操作する。記録の開始・保存・破棄は右A・右Bで行う |
| 補助者（安全係） | 非常停止スイッチの前に立ち、異常があれば即押す。操作者の周囲の安全を確認する |
| 記録係 | ワークを右トレイに置き直して蓋を開け、記録表（付録のワークシート）にエピソード番号と結果をつける |

10エピソードごとに操作者を交代すると、疲れによる操作の乱れを防げます。ただし操作の「やり方」はグループで統一してください（6-3）。

### 6-2. 1エピソードの流れ

1. 記録係が、3-5 の表に従って右トレイにワークを置き、蓋を止めブロックまで開ける
2. 操作者が **左X** で2台を初期位置へ戻す（初期位置からの開始を毎回そろえる）
3. 操作者が **右A** で記録開始（PC に「● 記録開始」）
4. ロボットB：右トレイの上へ → 下げる → 右トリガーで吸気 ON（ワークを吸着） → 持ち上げ → 箱の上へ → 箱の中へ下ろす → 右トリガーで吸気 OFF（放す） → 上へ退避
5. ロボットA：Bが退避したのを見てから、蓋の左外側へ → **手首を右へひねってノズルを傾ける** → 蓋の上のほうにノズルの側面を当てる → 右へ押して蓋を倒す → 閉じた蓋の上を軽く押さえる → 上へ退避 → **手首を戻して Rx を元の角度に戻す**
6. ロボットBの吸気・ロボットAの DO1：止まってからトリガーを1回引いて切り替える（毎回同じタイミングで）
7. 2台を初期位置付近まで戻して、**右A** で保存（PC に「保存しました → 合計 ○ エピソード」）
8. 失敗したとき（ワークを落とした、蓋が閉じなかった、箱を動かした等）は **右B** で破棄し、1 からやり直す

> 記録中は左X（初期位置へ）が効きません。初期位置への移動は記録の前後に行います。

### 6-3. よいデータを集める5つのコツ

| コツ | 理由 |
| --- | --- |
| 毎回同じ順番・同じ経路で動かす | 人によって経路がばらばらだと、AIが「平均」の中途半端な動きを覚えてしまう |
| ゆっくり・なめらかに動かす | 速い動きや急停止は 10 fps の記録では粗くなり、まねしにくい |
| 手首は「傾ける」「戻す」をはっきり分ける | 位置を動かしながら少しずつひねると、Rx の変化が小さくぶれた動きになり、AIが傾けるタイミングを学びにくい。上空で止まって傾け、傾けたまま蓋に近づく |
| 傾ける角度を毎回そろえる（約 20〜30°） | 角度がばらばらだと、AIの出す角度が中途半端になり蓋に当たらない |
| つかむ・放す前後で一瞬止める | 「止まってから 吸気・DO1 を切り替える」区切りがはっきりし、タイミングを学びやすい |
| 失敗は保存しない（右Bで破棄） | 失敗した動きもお手本として学習されてしまう |
| 途中で「やり直し」の動きを入れない | 位置合わせを何度もやり直すと、迷う動きを覚える。大きくずれたら破棄する |

### 6-4. 記録の終了と再開

- PC でキー `q` を押すと、データセットを正しく閉じて終了します（`ds.finalize()`）。**コマンドプロンプトの×ボタンで閉じないでください。** 最後のファイルが壊れることがあります
- 次の授業で続きを記録するときは、もう一度 `python record_dataset.py` を起動します。既存のデータセットに追加され、エピソード番号も続きから始まります

#### 最初から記録し直すには

`record_dataset.py` は、データセットのフォルダ（`data\e6_box_lid`）があれば続きから追加し、なければ新しく作ります。最初からやり直すときは、このフォルダを別の名前にしてから起動します。

**方法1：今のデータを別名で残す（おすすめ）**

```powershell
cd C:\quest_dobot
Rename-Item data\e6_box_lid e6_box_lid_old1
python record_dataset.py
```

古いデータは `data\e6_box_lid_old1` に残るので、あとで見比べたり戻したりできます。2回目以降は `old2`、`old3` のように名前を変えます。

**方法2：今のデータを完全に消す**

```powershell
cd C:\quest_dobot
Remove-Item -Recurse -Force data\e6_box_lid
python record_dataset.py
```

消したデータは元に戻せません。

- Quest のページは開いたままでよい。プログラムを起動し直したら、ページを再読み込みして「開始」を押す
- Colab で学習済みの場合、Google ドライブには古いデータのコピーが残っている。新しいデータで学習するときは、ノートブックでデータを入れ直す
- 一部の失敗したエピソードだけを消すなら、全部やり直さなくても `lerobot-edit-dataset` で削除できる（付録A データセットの編集）

### 6-5. 収集量の目安

| エピソード数 | 期待できる結果 |
| --- | --- |
| 20 | 動き出すが、位置ずれに弱い。まず全体の流れを試すには十分 |
| 50 | 枠内のばらつきに対応し始める（この教材の標準） |
| 100 以上 | 成功率が安定する。ワークの位置を広くばらつかせるほど必要数が増える |

## 第7章 データの確認と再生

学習に送る前に、データの中身を数字・グラフ・実機の再生の3つで確かめます。ここで不良なエピソードを見つけると、学習の失敗を防げます。

### 7-1. 数字で確認する

```bat
python check_dataset.py 0
```

表示例と見るポイントです。

```
エピソード数 : 50
総フレーム数 : 14820（24.7 分）
fps          : 10
各エピソードの長さ [秒]:
  0:31  1:28  2:30  3:29  4:33 …
各値の範囲（最小 〜 最大）:
  A.x        状態    210.4 〜    365.2   行動    208.9 〜    366.0
  …
```

| 見る項目 | 正常 | 異常のときの対処 |
| --- | --- | --- |
| エピソード数 | 記録表の数と一致 | 足りなければ追加で記録する |
| 各エピソードの長さ | ほぼそろっている（±30% 程度） | 極端に短い・長いものは失敗の可能性。7-2 で確認する |
| 各値の範囲 | `config.py` の `limits` の内側 | 範囲外の値があれば記録時の不具合。そのエピソードを除く（付録A） |
| `A.``suction・B.do1` | 最小 0.0、最大 1.0 | A.do1 は蓋閉めで DO1 を使わなければ常に 0 でよい。B.suction はワークを吸着している区間だけ 1 になる。最大が 0 のままなら吸気を一度も ON にしていない |
| A.rx・B.rx | −40〜40（RX\_RANGE）の内側。A.rx は蓋を押すときに 20〜30 程度まで変化する | ほぼ 0 のまま変化しない → 手首ひねりが記録されていない（index.html の差し替えを確認）。開始時から大きくずれた値になっている → ROT\_REF の設定ミス |

### 7-2. グラフと画像で確認する

`dataset_check` フォルダに次のファイルができます。

| ファイル | 内容 |
| --- | --- |
| `episode_000_plot.png` | 10個の値（左列=ロボットA、右列=ロボットB。上から X・Y・Z・Rx・DO1／吸気）の時間変化。青=状態（実際の位置）、橙の破線=行動（目標） |
| `episode_000_start_top.jpg` など | 開始・中間・終了時点の真上カメラ・正面カメラの画像 |

グラフでは、次の3点を確かめます。

- 青の線が橙の破線を少し遅れて追いかけている（指令どおりロボットが動いた）
- ロボットBの吸気が OFF になってワークを放したあとに、ロボットAの Rx が変化し始めている（作業の順番）
- ロボットAの Rx が「傾ける → 保つ → 戻す」の台形の形になっている（傾けたまま蓋を押した）
- ロボットBの Z がつかむ前に下がり、直後に上がっている

### 7-3. 実機で再生する

記録した「行動」だけを使って、2台をもう一度動かします。再生で同じ作業ができれば、データは正しく記録されています。

```bat
python replay_episode.py 0
```

1. 2台が初期位置へ移動する
2. ワークと蓋を、エピソード0を記録したときと同じ位置・角度にする
3. Enter で再生開始。中断はキー `q`

> 再生は `AUTO_MAX_SPEED`（80 mm/s）で速度を制限しています。記録時に速く動かした部分は、少し遅れて追従します。

### 7-4. 自己チェック

- [ ] エピソード数が記録表と一致した
- [ ] 3つ以上のエピソードでグラフを確認し、B の吸気 → A の Rx の順に変化していた
- [ ] 1つ以上のエピソードを実機で再生し、ワークを箱に入れて蓋を閉じられた

## 第8章 Google Colab で学習

データセットを zip にして Google ドライブ経由で Colab に送り、GPU で ACT を学習させます。T4 GPU で 30,000 ステップ、2〜3時間が目安です（データ量や Colab の混雑で変わります）。

> PC の CPU で学習すると、1ステップに数秒かかり、30,000 ステップには数日かかります。このため学習だけを Colab で行います。

### 8-1. データセットを zip にしてドライブへ送る

1. 記録プログラムをキー `q` で終了していることを確認する
2. エクスプローラーで `C:\quest_dobot\data` を開く
3. `e6_box_lid` フォルダを右クリック →「圧縮先」→「ZIP ファイル」（Windows 10 は「送る」→「圧縮（zip 形式）フォルダー」）
4. できた `e6_box_lid.zip` を、Google ドライブの `マイドライブ/lerobot/` フォルダにアップロードする（フォルダがなければ作る）
5. `train_act_colab.ipynb` もドライブにアップロードし、ダブルクリック →「Google Colaboratory で開く」

### 8-2. ノートブックの実行

最初に「ランタイム」→「ランタイムのタイプを変更」→「T4 GPU」を選び、上から順にセルを実行します。

| セル | 内容 | 確認すること |
| --- | --- | --- |
| 1. GPU の確認 | `nvidia-smi` | `Tesla T4` などが表示される |
| 2. ドライブの接続 | ドライブをマウントし、フォルダ名を設定 | `e6_box_lid.zip` が一覧に出る |
| 3. インストール | `lerobot[training,dataset]==0.6.1` | PC と同じ `0.6.1` が表示される |
| 4. データセット展開 | zip を展開して中身を表示 | エピソード数・fps=10 が PC と一致する |
| 5. 自動バックアップ | 5分ごとに最新チェックポイントをドライブへコピー | 「バックアップを開始しました」 |
| 6. 学習 | `lerobot-train` を実行 | `loss` が下がっていく |
| 7. モデル保存 | 学習済みモデルを `act_e6_model.zip` にしてドライブへ | zip の大きさ（約200MB）が表示される |

学習のコマンド（セル6）の主な設定は次のとおりです。

```bash
lerobot-train \
  --dataset.repo_id=local/e6_box_lid \
  --dataset.root=/content/data/e6_box_lid \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/content/outputs/act_e6 \
  --job_name=act_e6 \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --num_workers=2 \
  --wandb.enable=false
```

| 設定 | 値 | 意味 |
| --- | --- | --- |
| `policy.type` | act | 方策の種類 |
| `policy.chunk_size` | 50 | 1回に予測する行動の数（10 fps で5秒分） |
| `policy.n_action_steps` | 10 | 予測のうち実行する数（推論時は `lerobot_config.py` の値が優先） |
| `batch_size` | 8 | 1回の学習で使うフレーム数。GPU メモリ不足のエラーが出たら 4 に下げる |
| `steps` | 30000 | 学習の回数。少ないと動きが粗く、多すぎると時間がかかる |
| `save_freq` | 5000 | チェックポイントを保存する間隔 |

### 8-3. 学習の進み方を読む

ログに `step:1000 … loss:0.452 … l1_loss:0.21` のように表示されます。`l1_loss` は「お手本の行動とAIの予測の差」です。最初は 0.8 前後から始まり、学習が進むと下がっていきます。下がり方がほぼ止まれば、学習が落ち着いたと判断できます。

### 8-4. Colab が切断されたとき

無料版の Colab は、長時間の実行や操作がない状態で切断されることがあります。セル5のバックアップのおかげで、ドライブの `lerobot/act_e6_backup` に最大5分前までの途中経過が残っています。

1. ノートブックを開き直し、セル1〜4を実行する（5・6は実行しない）
2. セル「6'」の2つを実行する（バックアップを戻してから、続きから学習）

### 8-5. 学習済みモデルを PC に配置する

1. ドライブの `lerobot/act_e6_model.zip` を PC にダウンロードする
2. `C:\quest_dobot\models\act_e6\` を作り、そこに展開する
3. 次の構成になっていることを確認する

```
C:\quest_dobot\models\act_e6\pretrained_model\
 ├─ config.json
 ├─ model.safetensors
 ├─ policy_preprocessor.json
 ├─ policy_preprocessor_step_3_normalizer_processor.safetensors
 ├─ policy_postprocessor.json
 ├─ policy_postprocessor_step_0_unnormalizer_processor.safetensors
 └─ train_config.json
```

> `policy_preprocessor` と `policy_postprocessor` は、位置 \[mm\] などの値をAI用に正規化・逆正規化するための情報です。これがないと正しく動かないので、`pretrained_model` フォルダは丸ごと配置してください。

## 第9章 学習済みモデルで自律動作

PC の CPU で学習済みモデルを動かし、カメラ映像から2台を自動で動かします。Quest は使いません。自律動作は人が操作しないため、遠隔操作以上に安全確認を徹底します。

### 9-1. 開始前の安全確認

- [ ] 補助者が非常停止スイッチ（2台分）の前に立っている
- [ ] ロボットの動作範囲に人の手や物がない
- [ ] `config.py` の `limits` が記録時と同じ（AIの出力もこの範囲に制限される）
- [ ] `lerobot_config.py` の `RX_CENTER`・`RX_RANGE` が記録時と同じ（Rx もこの範囲に制限される）
- [ ] `AUTO_MAX_SPEED` が 80 以下、`AUTO_RX_MAX_SPEED` が 30 以下
- [ ] カメラ・照明・箱・トレイの位置と、蓋の開き角が記録時と同じ

### 9-2. 実行

```bat
cd C:\quest_dobot
.venv\Scripts\activate
python run_policy.py
```

1. 「モデルを読み込みます」のあと、カメラと2台に接続する
2. キー `h` で2台を初期位置へ移動する
3. 右トレイにワークを置き、蓋を止めブロックまで開ける
4. **スペース** で自律動作を開始する
5. 蓋が閉じたら **スペース** で一時停止し、`h` で初期位置へ戻す
6. ワークを右トレイに置き直し、蓋を開け直して 4 からくり返す。終了はキー `q`

| キー | 動作 |
| --- | --- |
| スペース | 開始 ／ 一時停止 |
| h | 一時停止して2台を初期位置へ |
| s | 一時停止して DO1・吸気とも OFF（ワークを落としてよいとき） |
| c | 一時停止してエラー解除・再イネーブル |
| q | 終了（DO1・吸気 OFF・切断） |

制限時間（`RUN_TIME_SEC` = 60秒）を過ぎると自動で一時停止します。

### 9-3. 安全のしくみ

AIの出力はそのままロボットに送らず、`ArmDriver` が次の制限をかけます。

| 制限 | 内容 |
| --- | --- |
| 動作範囲 | 目標位置を `config.py` の `limits` の内側に丸める |
| Rx の範囲 | 目標 Rx を `RX_CENTER` ± `RX_RANGE`（±40°）の内側に丸める |
| 速度 | 1周期（0.05秒）の移動量を `AUTO_MAX_SPEED` × 0.05 mm 以内、回転量を `AUTO_RX_MAX_SPEED` × 0.05° 以内にする |
| 姿勢 | 開始時の姿勢をベースの X 軸まわりに回して Rx を動かす（動かすのは位置 X, Y, Z と Rx だけ） |
| 時間 | `RUN_TIME_SEC` を過ぎたら一時停止 |

> これらはソフトウェアによる制限です。2台の衝突や想定外の動きは、非常停止スイッチで止めてください。

### 9-4. 推論の速さについて

ACT は1秒（10ステップ）ごとに、カメラ画像2枚から次の5秒分の行動を予測し直します。予測し直す瞬間に 0.2〜0.6 秒ほどかかり、PC 画面に `推論 ○○ ms（予測し直しのタイミング）` と表示されます。その間も `ArmDriver` は直前の目標へ動き続けます。

動きが止まりがちなときは、`lerobot_config.py` の `N_ACTION_STEPS` を 20 にすると、予測し直す回数が半分になります。ただし、ずれへの修正は遅くなります。

### 9-5. 結果の記録

1回ごとに、付録のワークシートに結果を記入します。成功の判定は次のとおりです。

| 判定 | 条件 |
| --- | --- |
| 成功 | ワークが箱の中にあり、蓋が閉じ、2台が退避した |
| 部分成功 | ロボットBはワークを箱に入れたが、ロボットAが蓋を閉じられなかった |
| 失敗 | ロボットBがワークをつかめない・箱の外に落とした、または非常停止した |

## 第10章 評価と改善・演習課題

10回試して成功率を数え、失敗の型からデータの足りない部分を見つけて追加収集します。この「評価 → データ追加 → 再学習」を1周回すことが、この教材のゴールです。

### 10-1. 成功率を測る

同じ条件で10回試し、成功・部分成功・失敗を数えます。ワークの置き方は 3-3 の表から毎回変えます。

| 指標 | 計算 |
| --- | --- |
| 成功率 | 成功の回数 ÷ 10 |
| ロボットBの成功率 | （成功＋部分成功）の回数 ÷ 10 |
| 平均時間 | 成功した回の、開始から2台が退避するまでの秒数の平均 |

### 10-2. 失敗の型と改善策

| 失敗の型 | 主な原因 | 改善策 |
| --- | --- | --- |
| Bがワークの真上で止まり、下がらない | 下げる動きのお手本が少ない／ばらつく | 下げる前に一瞬止める操作で10エピソード追加 |
| Bがワークの少し横に下りてつかめない | ワーク位置のばらつきが足りない | 失敗した位置付近にワークを置いて10エピソード追加 |
| Aがノズルを傾けない（Rx がほぼ 0 のまま） | 傾ける角度・タイミングがエピソードごとにばらばら | 「上空で止まって 25° 傾ける」をそろえて10エピソード追加 |
| Aが傾けるが蓋に届かない・空振りする | 蓋の開き角のばらつきに対応できていない | 失敗した開き角で10エピソード追加。正面カメラに蓋がはっきり写っているか確認 |
| Aが蓋を押しすぎて箱がずれる | 押さえる動きが強い／長い | 押さえる高さを毎回そろえる。箱の固定を強くする |
| AがBの退避前に動き出し、箱の上でぶつかりそう | Bの退避を待つお手本が少ない | Bが十分上がってからAを動かすエピソードを追加 |
| 照明や人影で動きがおかしくなる | 記録時と見え方が違う | カーテン・照明を固定。人が写らない位置に立つ |

データを追加したら、`record_dataset.py` で同じデータセットに記録を足し、第8章の学習を最初からやり直します（zip を作り直してアップロード）。

### 10-3. 設計の考え方：行動は「絶対位置」で記録する

この教材では、行動を「次に向かう目標位置（絶対座標 mm）」として記録しています。「前回からの移動量（差分）」で記録すると、ACT が予測を平均化したときに移動量が小さくなり、さらに誤差が積み重なって、ワークの高さまで下がりきらない現象が起きやすいためです（ROS 2 ＋ PS4 版の検証で実際に起きた現象です）。絶対位置なら、AIの予測が少しずれても、ずれがその場限りで済みます。

Rx も同じ考え方で、「`RX_CENTER` からのずれ（絶対角度）」を目標として記録しています。また、E6 の Rx は 180° と −180° が同じ向きなので、そのまま記録すると値が跳び、AIが「−180 と 180 の平均＝0°」のような誤った角度を出すことがあります。ずれで記録しているのはこのためです。

### 10-4. 演習課題

1. **データ量の比較**：最初の20エピソードだけで学習したモデルと、50エピソードで学習したモデルの成功率を比べ、表とグラフにまとめる（20エピソードだけのデータセットは付録Aの方法で作れる）
2. **カメラの役割**：正面カメラを外したデータセット（付録A）で学習し、蓋閉めの成功率がどう変わるか調べる。蓋の角度の判断にどちらのカメラが効いているかを考察する
3. **Rx の効果**：Rx を使わずに（手首をひねらずに）真上から蓋を押す方法で20エピソード記録・学習し、Rx ありの場合と成功率を比べる
4. **予測し直しの間隔**：`N_ACTION_STEPS` を 5・10・20 に変えて成功率と動きのなめらかさを比べる（再学習は不要）
5. **新しい課題**：「Aが蓋を開け、Bがワークを取り出す」逆の作業を設計し、データ収集から評価まで行う

### 10-5. 発表のまとめ方

| 項目 | 書くこと |
| --- | --- |
| 課題 | 何をさせたか（写真1枚） |
| データ | エピソード数、1エピソードの平均時間、ばらつきのつけ方 |
| 結果 | 成功率（改善前・改善後） |
| 失敗の分析 | 多かった失敗の型と、その原因の考察 |
| 改善 | 追加したデータの内容と、成功率の変化 |

## 付録

### 付録A データセットの編集

LeRobot 付属の `lerobot-edit-dataset` で、元のデータセットを残したまま編集済みのコピーを作れます。仮想環境を有効にして `C:\quest_dobot` で実行します。

**不良エピソードを除く**（例：エピソード 3 と 17 を除いて `e6_clean` を作る）

```bat
lerobot-edit-dataset --repo_id=local/e6_box_lid --root=data\e6_box_lid ^
  --new_repo_id=local/e6_clean --new_root=data\e6_clean ^
  --operation.type=delete_episodes --operation.episode_indices="[3,17]"
```

**最初の20エピソードだけにする**（演習1。20〜49 を除く）

```bat
lerobot-edit-dataset --repo_id=local/e6_box_lid --root=data\e6_box_lid ^
  --new_repo_id=local/e6_first20 --new_root=data\e6_first20 ^
  --operation.type=delete_episodes --operation.episode_indices="[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]"
```

**正面カメラを外す**（演習2）

```bat
lerobot-edit-dataset --repo_id=local/e6_box_lid --root=data\e6_box_lid ^
  --new_repo_id=local/e6_top_only --new_root=data\e6_top_only ^
  --operation.type=remove_feature --operation.feature_names="[observation.images.front]"
```

編集したデータセットで学習するときは、そのフォルダを zip にしてドライブに置き、ノートブックのセル2の `DATASET_NAME` をフォルダ名（例：`e6_clean`）に変えます。

### 付録B トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `ロボットA（192.168.5.1）に接続できません` | DobotStudio Pro・ROS 2（Docker）の bringup・`quest_dobot_teleop.py` のどれかが接続中。すべて終了する（5-2）。リモート制御モードが TCP/IP か確認 |
| `電源を入れます（20 秒待ちます）` のあとも動かない | PowerOn 後は約20秒待たないと有効化できない。キー `c` でエラー解除・再イネーブルする |
| `カメラ top（番号 0）を開けません` | 番号違い。`find_cameras.py` で確認して `lerobot_config.py` を直す。Zoom・Teams などカメラを使うアプリを終了する |
| `カメラ top の画像が更新されていません` | USB の帯域不足。2台を別の USB ポート（PC 本体側）に挿す |
| `注意: 記録周期に遅れています` が頻繁に出る | PC の負荷が高い。他のアプリを閉じる。改善しなければ `FPS` を 10 のまま、カメラ解像度を 320×240 に下げる（記録・学習・推論すべて同じ設定にする） |
| Quest の右A・右B・左X が反応しない | Quest のページが古い `index.html` のまま。ページを再読み込みし、操作表が出ることを確認 |
| `ModuleNotFoundError: No module named 'lerobot'` | 仮想環境が無効。`.venv\Scripts\activate` を実行してから起動する |
| 保存中にコマンドプロンプトを閉じてしまった | 最後のエピソードが壊れた可能性がある。`check_dataset.py` でエラーになる場合は、データセットのフォルダをバックアップから戻すか、付録Aで最後のエピソードを除く |
| Colab で `CUDA out of memory` | セル6の `--batch_size=8` を `4` に下げる |
| Colab で `FileNotFoundError`（データセット） | zip 内のフォルダ名と `DATASET_NAME` が一致しているか確認。`!ls /content/data` で中身を見る |
| `run_policy.py` でモデルを読み込めない | `models\act_e6\pretrained_model` の中に `config.json` があるか確認（フォルダが二重になっていないか）。PC と Colab の LeRobot のバージョンを `0.6.1` にそろえる |
| 自律動作でワークの上まで行くが下がらない | 10-2 の改善策を参照。下げる動作のデータを追加して再学習 |
| 自律動作でまったく違う方向へ動く | カメラ番号（top と front）が記録時と入れ替わっていないか確認 |
| 手首をひねっても Rx が変わらない | Quest のページが古い index.html のまま（向きが送られていない）。差し替えて再読み込みする。操作表に「手首をひねる」の行があれば新しいページ |
| Rx が逆向きに回る | lerobot\_config.py の RX\_SIGN のそのロボットを -1 にする |
| Rx を傾けるとエラー停止する（関節の限界・特異点） | RX\_RANGE を 30 → 20 と小さくする。初期姿勢（home\_joint）を、手首の関節に余裕のある姿勢に変える。キー c でエラー解除 |
| Rx の値が、ひねっていないのに大きな値で記録される | ROT\_REF が初期位置の姿勢と合っていない。5-5 の手順で設定し直し、記録したデータは作り直す |
| 傾けたノズルが箱や作業台に当たる | 傾けるとノズル先端の位置が横にずれる。上空で傾けてから近づく。limits の Z 下限を上げる |
| ロボットAの DO1・ロボットBの吸気が切り替わらない | PC に「\[A\] do1 ON / OFF」「\[B\] suction ON / OFF」は出るが機器が動かない → 制御盤 DO とツール DO の取り違え。DobotStudio Pro の IO 画面で端子を確認し、A は lerobot\_config.py の TOOLS、B は config.py の suction\_on / suction\_off を直す（ツール側なら ToolDOInstant(1,1) / (1,0)）。表示も出ない → Quest のページが古い（再読み込み） |
| DO1 の ON / OFF が逆 | TOOLS の A の on と off のコマンドを入れ替える。記録済みのデータは A.do1 の意味が逆になるため作り直す |
| 手首をひねっても Rx が変わらない（PC に「Rx +○°」が出ない） | ① Rx は record\_dataset.py でだけ動く（手順書の quest\_dobot\_teleop.py では動かない）。② PC に「Quest のページが古い版です」と出たら Quest Browser でページを再読み込みし、「開始」からやり直す。③ 起動時に「基準（ROT\_REF）から X 軸まわりに ○° 回っています」と出たら、初期位置で get\_pose.py を実行し、「ROT\_REF 用」の値を lerobot\_config.py の ROT\_REF に書く。④ ノズルはベース座標の X 軸まわりに回る。ひねる向きと回る向きが逆なら RX\_SIGN を -1 にする |
| PC に「Rx +○°」は出るがロボットが傾かない | PC に「NG: ServoP(...)」が出ていないか確認。出ていれば姿勢が限界に近いので RX\_RANGE を小さくするか、初期位置（home\_joint）を手首に余裕のある姿勢にする |
| 起動時に「WARNING: Could not load libtorchcodec」と長い表示が出る | エラーではなく警告。動画の読み書きは PyAV に自動で切り替わり、そのまま動く。表示を消すには仮想環境で pip uninstall -y torchcodec を実行する（Windows では FFmpeg の DLL がないため torchcodec を使えない） |
| Colab の学習セルで「ImportError: cannot import name 'FqnToConfig' from 'torchao.quantization'」と出て止まる | Colab に最初から入っている torchao と diffusers の版が合っていない。インストールのセルの後に !pip uninstall -y torchao を実行してから、学習のセルをもう一度実行する（ACT の学習には torchao を使わない）。配布のノートブックには、この1行を入れてある |

### 付録C 設定パラメータ一覧（`lerobot_config.py`）

| パラメータ | 初期値 | 説明 |
| --- | --- | --- |
| `DATASET_REPO_ID` | local/e6\_box\_lid | データセットの名前 |
| `DATASET_ROOT` | data\\e6\_box\_lid | データセットの保存先 |
| `TASK` | （英文） | 作業の説明文。記録と推論で同じにする |
| `FPS` | 10 | 記録・推論の周期 \[回/秒\]。途中で変えない |
| `EPISODE_TIME_SEC` | 60 | 1エピソードの最大時間 \[秒\] |
| `CAMERAS` | top=0, front=1 | カメラ番号と解像度 |
| `POLICY_PATH` | models\\act\_e6\\pretrained\_model | 学習済みモデルの場所 |
| `POLICY_DEVICE` | cpu | 推論に使う装置 |
| `N_ACTION_STEPS` | 10 | 予測し直すまでに実行する行動数 |
| `RUN_TIME_SEC` | 60 | 自律動作の制限時間 \[秒\] |
| `AUTO_MAX_SPEED` | 80 | 自律動作・再生の最大速度 \[mm/s\] |
| `SUCTION_THRESHOLD` | 0.5 | AIの出力を DO1・吸気の ON と判断するしきい値 |
| ROT\_REF | A=None, B=None | Rx の基準姿勢 \[Rx, Ry, Rz\]。初期位置で get\_pose.py の「ROT\_REF 用」の値を書く（None は起動時の姿勢） |
| RX\_RANGE | 40 | 基準姿勢から傾けてよい角度 \[deg\]（左右とも） |
| RX\_SIGN | A=1, B=1 | 手首のひねりと Rx の向きが逆なら -1 |
| RX\_SCALE | 1.0 | 手首のひねり角に対する Rx の倍率（遠隔操作時） |
| RX\_MAX\_SPEED | 45 | 遠隔操作時の Rx 最大回転速度 \[deg/s\] |
| AUTO\_RX\_MAX\_SPEED | 30 | 自律動作・再生時の Rx 最大回転速度 \[deg/s\] |
| TOOLS | A=DO1（toggle）、B=吸気（toggle） | トリガーで操作する出力。hold\_off=引いている間だけ OFF、toggle=引くたびに切替。A のコマンドは DOInstant(1,1) / DOInstant(1,0)、B は None（config.py の suction\_on / suction\_off を使う） |

ロボットの IP・初期位置・動作範囲・吸気コマンド（ロボットB）は、手順書の `config.py` の設定がそのまま使われます。

### 付録D 記録ワークシート

**データ収集記録**（班名：＿＿＿＿　日付：＿＿＿＿）

| エピソード | ワーク位置・蓋の角度（3-5の表） | 操作者 | 結果（保存／破棄） | メモ |
| --- | --- | --- | --- | --- |
| 0 | 中央 |  |  |  |
| 1 | 中央 |  |  |  |
| 2 | 中央 |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |

**自律動作の評価**（モデル名：＿＿＿＿　学習データ：＿＿エピソード　ステップ数：＿＿＿＿）

| 回 | ワーク位置 | 結果（成功／部分成功／失敗） | 時間 \[秒\] | 失敗の型（10-2） |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| 6 |  |  |  |  |
| 7 |  |  |  |  |
| 8 |  |  |  |  |
| 9 |  |  |  |  |
| 10 |  |  |  |  |

成功率：＿＿／10　　ロボットBの成功率：＿＿／10　　平均時間：＿＿秒

### 付録E 参考資料

| 資料 | 内容 |
| --- | --- |
| [LeRobot ドキュメント](https://huggingface.co/docs/lerobot) | LeRobot の公式ドキュメント |
| [Bring Your Own Hardware（LeRobot）](https://huggingface.co/docs/lerobot/integrate_hardware) | 独自ロボットを LeRobot に組み込む方法 |
| [LeRobot（GitHub）](https://github.com/huggingface/lerobot) | ソースコード（`lerobot-train`・`lerobot-edit-dataset` など） |
| [DOBOT TCP/IP Remote Control Interface Guide V4.6.6](https://a.storyblok.com/f/298593/x/ce7fbaa06a/dobot-tcp_ip-remote-control-interface-guide-v4-6-6_20260527_en.pdf) | E6 の TCP/IP コマンド仕様（ServoP、PowerOn、RobotMode など） |
| Meta Quest Touch Plus による DOBOT Magician E6 ×2台 遠隔操作 手順書 | この教材の前提となる遠隔操作の手順書 |
| DOBOT E6 を ROS 2 で動かす手順書 | ROS 2（Docker）版の環境構築。PowerOn 後の待ち時間などの注意点を参照 |

### 免責事項

本記事・本ソフトウェアの使用によって生じた、データの消失、利益の損失、デバイスの故障（正常に起動しなくなる等）、またはその他の直接的、間接的、特別、偶発的、あるいは派生的な損害について、著作者は一切の責任を負いません。 すべての操作は、利用者ご自身の責任において行ってください。
