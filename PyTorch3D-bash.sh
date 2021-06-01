sudo apt install python3.7 -y
sudo apt install python3-pip -y

sudo apt-get update -y
sudo apt-get upgrade -y linux-aws
sudo apt  install awscli -y
sudo apt-get install -y gcc make linux-headers-$(uname -r)
cat << EOF | sudo tee --append /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF
sudo chmod 777 /etc/default/grub
sudo echo 'GRUB_CMDLINE_LINUX="rdblacklist=nouveau"' >> /etc/default/grub
sudo chmod 400 /etc/default/grub
sudo update-grub

aws configure set aws_access_key_id XXXXXX
aws configure set aws_secret_access_key XXXXXX


aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
aws s3 ls --recursive s3://ec2-linux-nvidia-drivers/
chmod +x NVIDIA-Linux-x86_64*.run
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
echo "PATH=/usr/local/cuda-10.2/bin:$PATH" >> .bashrc
echo "LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH" >> .bashrc
nvcc --version

python3.7 -m pip install torch==1.7
python3.7 -m pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt170/download.html

python3.7 -m pip install -r requirements.txt