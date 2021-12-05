export PYTHONPATH=$PYTHONPATH:`pwd`

python -m bdd100k.eval.run -t seg_track -g ../data/bdd/labels/seg_track_20/bitmasks/val -r ../converted_results/seg_track
