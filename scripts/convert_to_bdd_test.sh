export PYTHONPATH=$PYTHONPATH:`pwd`

python ./tools/to_bdd100k.py configs/segtrack-frcnn_r50_fpn_12e_bdd10k_fixed_pcan.py --res eval_result_pcan_test.pkl --task seg_track --bdd-dir converted_results_test/ --nproc 2

