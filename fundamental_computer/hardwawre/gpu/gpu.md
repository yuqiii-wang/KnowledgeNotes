# GPU

## Intro

## How to Use

### Quick PyTorch Install

Goto `https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local` to install driver.

Download and install Anaconda (in China, goto `https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/`).

Goto `https://pytorch.org/get-started/locally/` to install pytorch `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (Sometimes `--upgrade --force-reinstall` is required if in python `torch.cuda.is_available()` says no after repeated reinstalls).

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

## Nvidia GPU and Ubuntu Monitor Display (HDMI Signal)

Incompatible Nvidia GPU driver versions can cause Ubuntu failed to detect HDMI signal (Monitor Display).
This happens happens when just having installed/updated Anaconda, CUDA, PyTorch, or other GPU related software.

Run below bash to re-install Nvidia driver. 

```bash
sudo ubuntu-drivers autoinstall

## Optional fix
sudo apt --fix-broken install
```

## OpenGL

OpenGL (Open Graphics Library) is a popular APIs for rendering 2D and 3D vector graphics,
managed by Khronos Group (The Khronos Group, Inc. is an open, non-profit, member-driven consortium of 170 organizations developing, publishing and maintaining royalty-free interoperability standards for 3D graphics, virtual reality, augmented reality, parallel computation, vision acceleration and machine learning).

<div style="display: flex; justify-content: center;">
      <img src="imgs/Linux_kernel_and_OpenGL_video_games.svg.png" width="70%" height="70%" alt="Linux_kernel_and_OpenGL_video_games.svg" />
</div>
</br>

OpenGL is the often the default acceleration program for Linux-like systems, such as gaming platform steam, browsers, and video players.

### OpenGL vs DirectX

Both openGL vs DirectX serve the same use scenarios of accelerating computing graphic tasks,
but DirectX is proprietary + non-open source owned by Microsoft.
Generally speaking, DirectX outperforms OpenGL in most graphic tasks.

|OpenGL|DirectX|
|-|-|
|Open source, cross-platform (Linux, Mac OS, and Windows)|Non-open source, only on Windows (and Microsoft platforms such as Xbox)|
|Generally worse in performance, such as no support for ray tracing computation facilitation (popular in gaming)|Generally Better in performance|