#!/usr/bin/env bash
repeat_times=1
CONFIG=pc/graph_level_pc
# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/${CONFIG}.yaml --repeat $repeat_times # node classification


# python main.py --cfg configs/pyg/example_link.yaml --repeat 3 # link prediction --mark_done
# python main.py --cfg configs/pyg/example_graph.yaml --repeat 3 # graph classification
