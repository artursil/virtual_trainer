#!/bin/bash
conda activate dev-env
python combined_learning_v8.py &
pid=$!
wait $pid
echo Process $pid finished.
python combined_learning_v9.py &
pid=$!
wait $pid
echo Process $pid finished.

