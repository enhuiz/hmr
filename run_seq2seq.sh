#!/bin/bash

config=$1

if [ -z $config ]; then
    echo "Please provide a json config file."
else
    opts=$(scripts/utils/parse_opts.py $config)
    opts=$(eval echo $opts)
    if [ -z "$opts" ]; then
        echo "No options provided."
    else
        python3 -u scripts/train_seq2seq.py $opts
    fi
fi
