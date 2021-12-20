# GPU

## Intro

## How to Use

### docker for nvidia

1. **Add nvidia docker repo key to apt**
```bash
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```
if got `connection refused`, you can manually download the key then
```bash
cat gpgkey | sudo apt-key add -
```

2. **To retrieve os name**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
```

3. **Write into apt update list**
```bash
curl -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
You can manually copy and paste into `/etc/apt/sources.list.d/nvidia-docker.list` from browser opening `https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list` (**remember to input your own os distribution name**)

If resolution to `nvidia.github.io` failed, you can manually add `185.199.109.153 nvidia.github.io` into `vim /etc/hosts`

4. **update apt list**

run `sudo apt update`

5. **install docker**

run `sudo apt-get install -y nvidia-docker2`