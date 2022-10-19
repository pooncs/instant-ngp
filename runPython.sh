basep=./data/nerf/shuttle4
savep=/testcase/
caseN=13
configN=base
tsteps=50000

if [[ -d $basep$savep$caseN ]]
then
    echo "$basep$savep$caseN exists on your filesystem."
else
    mkdir -v -p $basep$savep$caseN
fi

if [ $1 == "t" ]
then
    echo "Training data"
    
    nohup python3 ./scripts/run.py --mode nerf \
            --scene $basep \
            --n_steps $tsteps \
            --network $configN.json \
            --save_snapshot $basep$savep$caseN/$configN-$tsteps.msgpack \
            --screenshot_dir $basep$savep$caseN/ \
            --nadir $basep/jsons/nadir4.json \
            --screenshot_spp 64 \
            |& tee $basep$savep$caseN/$configN-$tsteps-tlog.txt
fi

if [ $1 == "r" ]
then
    echo "Rendering data"
    python3 ./scripts/run.py --mode nerf \
            --scene $basep \
            --network $configN.json \
            --load_snapshot $basep$savep$caseN/$configN-$tsteps.msgpack \
            --screenshot_transforms $basep/transforms.json \
            --screenshot_frames 7 \
            --screenshot_dir $basep$savep$caseN/ \
            --nadir $basep/jsons/nadir3.json \
            --screenshot_spp 64 \
            |& tee $basep$savep$caseN/$configN-$tsteps-rlog.txt
fi