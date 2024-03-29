# GLF-CR: SAR-Enhanced Cloud Removal with Global-Local Fusion

This repository contains the codes for the paper "GLF-CR: SAR-Enhanced Cloud Removal with Global-Local Fusion" 


If you use the codes for your research, please cite us accordingly:

```
@article{xu2022glf,
  title={GLF-CR: SAR-enhanced cloud removal with global--local fusion},
  author={Xu, Fang and Shi, Yilei and Ebel, Patrick and Yu, Lei and Xia, Gui-Song and Yang, Wen and Zhu, Xiao Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={192},
  pages={268--278},
  year={2022},
  publisher={Elsevier}
}
```

## Prerequisites & Installation

This code has been tested with CUDA 10.1 and Python 3.6.

```
conda create -n GLF-CR python=3.6
pip install torch==1.4.0 torchvision==0.5.0
pip install scipy
pip install rasterio
pip install timm==0.3.2

cd ./codes/FAC/kernelconv2d/
python setup.py clean
python setup.py install --user
```

## Get Started
You can download the pretrained model from [here](https://drive.google.com/file/d/11EYrrqLzlqrDgrJNgIW7IY0nSz_S5y9Z/view?usp=sharing) and put it in './cpkg'.

Use the following command to test the neural network:
```
python test_CR.py
```

## Credits

This code is based on the codes available in the [STFAN](https://github.com/sczhou/STFAN) repo, [slow-motion](https://github.com/MeiguangJin/slow-motion) and [SwinIR](https://github.com/JingyunLiang/SwinIR). I am grateful to the authors for making the original source code available.

## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to xufang@whu.edu.cn
