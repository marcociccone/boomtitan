
podman build -t cuda:12.8.1-cudnn-devel-ubuntu24.04 -f Dockerfile .
rm -rf $SCRATCH/ce-images/cuda1281.sqsh
enroot import -x mount -o $SCRATCH/ce-images/cuda1281.sqsh podman://cuda:12.8.1-cudnn-devel-ubuntu24.04


