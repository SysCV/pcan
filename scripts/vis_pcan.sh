export PYTHONPATH=$PYTHONPATH:`pwd`

python3 ./tools/test.py configs/segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_pcan.py ./pcan_pretrained_model.pth --out eval_pcan_results_val.pkl --eval 'segtrack' --show-dir ./vis_pcan_result_val/

