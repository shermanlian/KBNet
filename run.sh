#!/usr/bin/env bash

python main.py --n_GPUs 4 --lr 0.0002 --decay 100-200 --save kbnet --model KBNet --n_feats 128 --n_resblocks 8 --n_resgroups 5 --batch_size 32 --burst_size 8 --patch_size 256 --scale 4 --loss 1*L1

# python test.py --root ../test_set_0_1.6 --n_GPUs 1 --model KBNet --n_feats 128 --n_resblocks 8 --n_resgroups 5 --batch_size 64 --burst_size 2 --scale 4 --pre_train ../train_log/KBNet/real_models/kbnet/KBNetbest_epoch.pth
# python test.py --root ../test_set_1.6_3.2 --n_GPUs 1 --model KBNet --n_feats 128 --n_resblocks 8 --n_resgroups 5 --batch_size 64 --burst_size 2 --scale 4 --pre_train ../train_log/KBNet/real_models/kbnet/KBNetbest_epoch.pth
# python test.py --root ../test_set_3.2_4.8 --n_GPUs 1 --model KBNet --n_feats 128 --n_resblocks 8 --n_resgroups 5 --batch_size 64 --burst_size 2 --scale 4 --pre_train ../train_log/KBNet/real_models/kbnet/KBNetbest_epoch.pth
