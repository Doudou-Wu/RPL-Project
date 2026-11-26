
ssh wudoud@thorin.fmi.ch

cd ../../../tachyon/groups/gzenke/wudoud

ssh wudoud@vcl1092.fmi.ch

ssh wudoud@vcl1062.fmi.ch

source .venv/bin/activate

nvidia-smi
sinfo -N -o "%12N %10P %8T %6c %8m %25G %12e"

cd /tachyon/groups/gzenke/wudoud/projects/tlt12-ldm
