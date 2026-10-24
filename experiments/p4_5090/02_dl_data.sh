#!/bin/bash
# experiments/p4_5090/02_dl_data.sh — download all meta datasets (5090 self-download)
# Run on 5090: nohup setsid bash 02_dl_data.sh > /dev/null 2>&1 < /dev/null &
cd ~/ia_p4
~/ia_p4/venv/bin/python experiments/local_dl_meta.py --groups tv_small tv_large direct > logs/dl_data.log 2>&1
echo ALL_DATA_DONE >> logs/dl_data.log
