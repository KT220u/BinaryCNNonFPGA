# モデル
input : MNIST dataset, 28 * 28 = 784 bit<br>
convolutional_layer : 5 * 5 畳み込みフィルター１枚 <br>
hidden_layer : 128 ノード<br>
output_layer : 10 クラス分類<br>
<br>
かなり単純なモデルで、正解率は 90% 程度 <br>

# 各ファイルについて
layers.py : 各レイヤーの定義<br>
model.py : モデルの定義<br>
train.py : 学習<br>
test.py : テスト<br>
make_dat.py : Verilog のreadmemb 関数で用いるためのデータファイルを生成する。<br>

# 実行手順
train.py を実行して、model_weight.pth を生成 <br>
→ make_dat.py を実行して Verilogディレクトリ下にパラメータファイルを生成 <br>
