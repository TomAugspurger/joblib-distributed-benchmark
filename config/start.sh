#!/usr/bin/env bash
jupyter notebook --config=/work/config/jupyter_notebook_config.py --port=8888 /work &
jupyter lab      --config=/work/config/jupyter_notebook_config.py --port=8889 /work &
wait
