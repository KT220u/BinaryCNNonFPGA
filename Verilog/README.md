# 各ファイルについて
トップモジュール<br>
・Top.v<br>
モデル<br>
・InputLayer.v<br>
・ConvLayer.v<br>
・HiddenLayer.v<br>
・OutputLayer.v<br>
パラメータデータ (画像数 : 1024) <br>
・test_image.dat : 784 * 1024 <br>
・conv_weight.dat : 5 * 5 <br>
・hidden_weight.dat : (24 * 24 + 1) * 128 <br>
・output_weight.dat : (128 + 1) * 10<br>
# 実行方法
Pythonディレクトリ化でモデルの学習をし、パラメータデータを生成した上で iverilog.sh を実行すると a.out ができる
```
$ bash iverilog.sh 
$ ./a.out 
```
