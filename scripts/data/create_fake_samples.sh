#!/bin/bash

. ./scripts/utils/parse_options.sh

if [ $# != 1 ]; then
    echo "Usage: $0 <dir>"
    echo "main options (for others, see top of script file)"
    echo ""
    exit 1
fi

dir=$1

create_pngs() {
    echo $1
    mkdir -p $1
    cd $1
    touch $(seq -f "%g.png" $2 $3)
    cd -
}

create_features() {
    create_pngs $1/train 0 179
    create_pngs $1/dev 180 189
    create_pngs $1/test 190 199
}

cat_csv() {
    echo "id,$1"
    seq -f "%g,\"$2\"" $3 $4
}

create_annotations() {
    mkdir -p $1
    cat_csv $2 $3 0 179 >$1/train.csv
    cat_csv $2 $3 180 189 >$1/dev.csv
    cat_csv $2 $3 190 199 >$1/test.csv
}

create_mnist() {
    dir=$1/MNIST
    rm -rf $dir
    mkdir -p $dir
    create_features $dir/features/written
    create_features $dir/features/printed
    create_annotations $dir/annotations target 5
}

create_icfhr() {
    dir=$1/ICFHR
    rm -rf $dir
    mkdir -p $dir
    create_features $dir/features/written
    create_features $dir/features/printed
    create_annotations $dir/annotations target $(echo a^2+b^2=c^2)
}

create_mnist $dir
create_icfhr $dir
