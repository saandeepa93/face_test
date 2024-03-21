
sinfo -e -p <partition_name> -o "%9P %3c %.5D %6t " -t idle,mix
sinfo -o "%P %G %D %N"
ssh a11-02 'nvidia-smi -i 0,1,2,3,4,5,6,7'
sacct --format="Elapsed" -j 16332627

sinfo -p rra -o "%P  %3c %G %D %N %6t" -t idle,mix
sinfo -p rra_con2020 -o "%P  %3c %G %D %N %6t" -t idle,mix
sinfo -e -p rra -o "%9P %3c %.5D %6t " -t idle,mix
