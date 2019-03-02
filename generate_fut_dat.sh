#!/bin/bash

#HIST_SIZE=16 64 256 1024 4096 16384 61440
#NUM_HISTS=1 4 15 60 240 959 3835 15338 61350

# do not make these - slow to generate and not needed
#H:16384-61350
#H:61440-15338
#H:61440-61350

DATA_PATH=data/futhark

mkdir -p $DATA_PATH

for i in 16 64 256 1024 4096 16384 61440; do
    for j in 1 4 15 60 240 959 3835 15338 61350; do
        if [[  ($i == 16384 && $j == 61350) ||
               ($i == 61440 && ($j == 15338 || $j == 61350))
           ]]; then
            continue
        fi
        out="$DATA_PATH/$i-$j.dat"
        echo "Generating $out"
        if [ -f "$out" ]; then
            echo "$out already exists; not generating anew."
        else
            futhark dataset --binary --i32-bounds=0:$i \
                    -g [$j][$i]i32 > "$out"
        fi
    done
done
