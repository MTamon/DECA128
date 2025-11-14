pip install gdown > /dev/null 2>&1
rm ./data/deca_model.tar
gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O ./data/deca_model.tar

# デモスクリプトの実行
python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True