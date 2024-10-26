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
    # Install Flask
    pip install flask
    
    # Install Basicsr
    pip install basicsr

    # Install Facexlib
    pip install facexlib

    # Install Requirments
    pip install -r requirements.txt

    # Setup
    python setup.py develop
    
    # Install Real-ESRGAN
    pip install realesrgan
    ```
5. For one click run
    Create a new GFPGAN.bat Batch File on your GFPGAN directory using the script given below:

    ```console
    @echo off

    :: Activate the conda environment for GFPGAN
    CALL "C:\ProgramData\miniconda3\Scripts\activate.bat" GFPGAN

    :: Navigate to the GFPGAN directory (Change path accroding to yourself)
    cd /D C:\Users\<your_username>\Desktop\AI\GFPGAN-Web-Interface
    
    :: Run the GFPGAN web interface script
    python GFPGAN.py   
    ```

## :zap: Quick Inference

We take the v1.4 version.

Download pre-trained models: [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)
Put them into experiments/pretrained_models

**Web Inference!**

```bash
python GFPGAN.py
```
Go to [localhost:5000](http://127.0.0.1:5000)