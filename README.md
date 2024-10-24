## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Install Anaconda or Miniconda Globally (Check the 'All Users' option from the Anaconda or Miniconda installation window)
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

1. Create conda environmetnt

    ```bash
    conda create -n GFPGAN python=3.7
    conda activate GFPGAN
    ```
2. Install PyTorch

    ```bash
    # For NVIDIA GPU
    conda install pytorch torchvision torchaudio pytorch-cuda=<your_cuda_version> -c pytorch -c nvidia

    # For CPU
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

3. Clone repo

    ```bash
    git clone https://github.com/Md-Siam-Mia-Code/GFPGAN-Web-Interface.git
    cd GFPGAN-Web-Interface
    ```

4. Install dependent packages

    ```bash
    pip install basicsr
    pip install facexlib
    pip install -r requirements.txt
    python setup.py develop
    pip install realesrgan
    ```
5. For one click run
    Create a new shortcut on your GFPGAN directory using the script given below:

    ```console
    @echo off

    :: Activate the conda environment for GFPGAN
    CALL "C:\ProgramData\miniconda3\Scripts\activate.bat" GFPGAN

    :: Navigate to the GFPGAN directory (Change path accroding to yourself)
    cd /D C:\Users\<your_username>\Desktop\AI\GFPGAN
    
    :: Run the GFPGAN web interface script
    python web_interface_gfpgan.py   
    ```

## :zap: Quick Inference

We take the v1.4 version.

Download pre-trained models: [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P experiments/pretrained_models
```
**Web Inference!**

```bash
python web_interface_gfpgan.py
```
Go to [localhost:5000](http://127.0.0.1:5000)

**Console Inference!**

```bash
python inference_gfpgan.py -i Input -o Output -v 1.4 -s 2
```

```console
Usage: python inference_gfpgan.py -i Input -o Output -v 1.4 -s 2 [options]...

  -h                   show this help
  -i input             Input image or folder. Default: inputs/whole_imgs
  -o output            Output folder. Default: results
  -v version           GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3
  -s upscale           The final upsampling scale of the image. Default: 2
  -bg_upsampler        background upsampler. Default: realesrgan
  -bg_tile             Tile size for background sampler, 0 for no tile during testing. Default: 400
  -suffix              Suffix of the restored faces
  -only_center_face    Only restore the center face
  -aligned             Input are aligned faces
  -ext                 Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```
