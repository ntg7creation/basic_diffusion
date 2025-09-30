# basic_diffusion
how to train

python train_barebones_diffusion.py --config configs/barebones.yaml


how to infer 
python infer_barebones.py --config configs/barebones.yaml \
                          --ckpt save/barebones/model_epoch_1.pt \
                          --text "wave your hand"
