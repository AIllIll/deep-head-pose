## 实验环境
ubuntu18
## 克隆项目
```
git clone https://github.com/AIllIll/deep-head-pose.git
```
## 下载模型
地址：
```
https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR
```
## 项目环境
进入项目目录
```bash
cd deep-head-pose
```
新建conda环境并安装依赖
```
conda create -n deep-head-pose python=3.8
```
```
conda activate deep-head-pose
```
```
conda install pytorch
```
```
pip install opencv-python matplotlib torchvision scikit-image dlib pandas tensorflow
```

## 运行
连接摄像头后
```bash
python code/play.py --snapshot ./model/hopenet_robust_alpha1.pkl --face_model ./mmod_human_face_detector.dat --video 0
```