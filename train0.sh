


python train_MSA.py --token_len 50 --device 0 --epochs 61 --batch_size 64 --vision_feats OpenFace --audio_feats Librosa --datasetName MSA --model_name Trinity --dataPath none --note Trinity_MOSEI
python train_Action_cls.py --token_len 50 --device 0 --epochs 30 --batch_size 64 --vision_feats resnet2plus1D --audio_feats librosa --datasetName AVE --model_name Trinity --dataPath ./datasets/AVE_rms_k16_last.pkl --note Trinity_AVE
python train_Action_cls.py --token_len 50 --device 0 --epochs 20 --batch_size 64 --vision_feats resnet2plus1D --audio_feats librosa --datasetName UCF51 --model_name Trinity --dataPath ./datasets/UCF51_rms_k16_last.pkl --note Trinity_UCF51
python train_Action_cls.py --token_len 50 --device 0 --epochs 15 --batch_size 64 --vision_feats resnet2plus1D --audio_feats librosa --datasetName KS --model_name Trinity --dataPath ./datasets/KS_rms_k16_last.pkl --note Trinity_KS

python test_Action_cls.py  --token_len 50 --device 0 --batch_size 64 --vision_feats resnet2plus1D --audio_feats librosa --model_name Trinity
python test_MSA.py --token_len 50 --device 0 --batch_size 64 --vision_feats OpenFace --audio_feats Librosa --datasetName MSA --model_name Trinity


