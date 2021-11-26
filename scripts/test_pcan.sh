export PYTHONPATH=$PYTHONPATH:`pwd`

bash ./tools/dist_test.sh configs/segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_refine_pcan.py pcan_training_result_4gpu/epoch_12.pth 4 --out eval_result_pcan_test.pkl --eval 'segtrack'

