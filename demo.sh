pip install gdown > /dev/null 2>&1
# deca_model.tar のダウンロード (ファイルが存在しない場合のみ)
if [ ! -f ./data/deca_model.tar ]; then
  gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O ./data/deca_model.tar
fi

# デモスクリプトの実行
python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True