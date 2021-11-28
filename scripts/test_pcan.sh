export PYTHONPATH=$PYTHONPATH:`pwd`

bash ./tools/dist_test.sh configs/segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_pcan.py ./pcan_pretrained_model.pth 4 --out eval_result_pcan_test.pkl --eval 'segtrack'

