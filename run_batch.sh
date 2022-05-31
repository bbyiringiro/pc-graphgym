#!/usr/bin/env bash

CONFIG=pc/graph_level_pc
GRID=exp_pc
REPEAT=1
MAX_JOBS=8
SLEEP=1
MAIN=main
result_dir=results

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir configs
# python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# # rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# # rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python agg_batch.py --dir ${result_dir}/${CONFIG}_grid_${GRID}
