mkdir -p embeddings
mkdir -p pretrained/frontend


### VAD/OSD
wget https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin -P pretrained/frontend


### clova
mkdir -p embeddings/clova
git clone https://github.com/clovaai/voxceleb_trainer.git embeddings/clova/voxceleb_trainer
mkdir -p pretrained/clova
wget http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model -P pretrained/clova
# remove nested git repository
rm -rf embeddings/clova/voxceleb_trainer/.git*
# fix the original project to prevent import errors
cat embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py | sed -e "s/from models.ResNetBlocks import/from ..models.ResNetBlocks import/" > embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt
mv embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py
cat embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py | sed -e "s/from utils import PreEmphasis/from ..utils import PreEmphasis/" > embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt
mv embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py


### speechbrain
mkdir -p pretrained/speechbrain
wget https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt -P pretrained/speechbrain
wget https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/raw/main/hyperparams.yaml -P pretrained/speechbrain


### brno
mkdir -p pretrained/brno
wget https://data-tx.oss-cn-hangzhou.aliyuncs.com/AISHELL-4-Code/sd-part.zip
unzip sd-part.zip -d pretrained/brno
cat pretrained/brno/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip* > pretrained/brno/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip
unzip pretrained/brno/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip -d pretrained/brno/VBx/models/ResNet101_16kHz/nnet
mv pretrained/brno/VBx/models/ResNet101_16kHz pretrained/brno
rm -r pretrained/brno/VBx
rm sd-part.zip

