#!/bin/bash
set -e

stage=0

. ./scripts/utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <src> <dst>"
    echo "main options (for others, see top of script file)"
    echo "  --stage "
    exit 1
fi

src=$1
dst=$2

srcdir=${src%.zip}/Task1_and_Task2/

if [ $stage -le 0 ]; then
    inflat() {
        local src=$1
        echo "Inflating $src"
        local dir=$(dirname $src)
        unzip -n $src -d $dir >/dev/null
    }

    echo "Unzipping ..."

    inflat $src

    for zp in $srcdir/Task1_onlineRec/MainTask_formula/*.zip; do
        inflat $zp
    done

    for zp in $srcdir/Task2_offlineRec/MainTask_formula/*.zip; do
        inflat $zp
    done
fi

if [ $stage -le 1 ]; then
    extract_math() {
        inkml=$1
        math=$(grep -m1 'truth' $inkml)
        math=${math#*>}
        math=${math%<*}
        math=${math//$/}
        # &cmd; -> \cmd
        echo $math | perl -pe 's/&(.+?);/\\\1 /g'
    }

    annotate_helper() {
        local src=$1
        local dst=$2

        id=$(basename $src .inkml)
        math=$(extract_math $src)

        echo \"$id\",\"$math\" >>$dst
    }

    annotate() {
        local task=$1
        local dst=$2

        if [ -f $dst ]; then
            echo "$0: $dst exists, skip annotating."
        else
            mkdir -p $(dirname $dst)

            >$dst
            for file in $(find $srcdir/Task1_onlineRec/ -path "*/$task/*.inkml"); do
                annotate_helper $file $dst
            done
        fi
    }

    echo "$0: Annotating ..."
    annotate Train $dst/annotations/train.csv
    annotate valid $dst/annotations/dev.csv
    annotate Test $dst/annotations/test.csv
fi

if [ $stage -le 2 ]; then
    echo "Rendering stroke ..."
    cd $srcdir/Task2_offlineRec/
    ./ImgGenerator
    cd - 1>/dev/null

    wdir=$dst/features/written
    mkdir -p $wdir/{train,dev,test}/

    find $srcdir/Task2_offlineRec/ -path "*/Train/*.png" -exec cp {} $wdir/train/ \;
    find $srcdir/Task2_offlineRec/ -path "*/valid/*.png" -exec cp {} $wdir/dev/ \;
    find $srcdir/Task2_offlineRec/ -path "*/test/*.png" -exec cp {} $wdir/test/ \;
fi

if [ $stage -le 3 ]; then
    echo "Rendering latex ..."

    pdir=$dst/features/printed
    mkdir -p $pdir

    node scripts/data/math2png.js $dst/annotations/train.csv $pdir/train
    node scripts/data/math2png.js $dst/annotations/dev.csv $pdir/dev
    node scripts/data/math2png.js $dst/annotations/test.csv $pdir/test
fi

echo "done."
