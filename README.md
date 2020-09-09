# Inference_benchmark


### 一、环境准备  && 测试

#### 1. Paddle
Paddle官方发布了cuda10的docker镜像，首先拉取images:
```bash
docker images pull hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7
```

根据images，构建容器，并进入此容器
```bash
sudo docker run --name XXX --net=host -v $PWD:/workspace -it hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  /bin/bash
```

克隆paddle的仓库，编译预测库lib文件
```
git clone https://github.com/PaddlePaddle/Paddle.git

cd Paddle
mkdir build_infer

touch re_build.sh
```

在创建的`re_build.sh`里，添加如下内容：
```
#!/bin/bash

cmake .. -DWITH_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_MKL=ON \
        -DWITH_GPU=ON \
        -DON_INFER=ON \
        -DWITH_CONTRIB=OFF \
        -DWITH_XBYAK=OFF \
        -DWITH_PATHON=OFF \
        -DWITH_MKLDNN=OFF

make -j12
make inference_lib_dist -j12
```

执行脚本，编译预测库，最终会在`build_infer`目录下生成一个`fluid_inference_install_dir`目录，包含了预测库文件，后续用于配置此路径。

然后克隆此仓库
```
git clone https://github.com/Aurelius84/Inference_benchmark.git
cd paddle/inference
```

inference目录下有一个`image_classification.cc`，是resnet50/mobileNetv1的预测样例代码，可以编译测试：
```
./re_build.sh
```

执行完毕后，会在当前目录生成一个`image_classification`可执行程序。如果本地有resnet的预测模型，则可以执行预测latency的评测：
```bash
./test_latency 16 ../static/resnet50

# 其中 16为batch_size, 后面为模型路径
```

#### 2. Torch

Torch官方也提供了cuda10的镜像，首先拉取images:
```bash
docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
```

根据拉取的镜像，创建容器：
```bash
sudo docker run --name XXX --net=host -v $PWD:/workspace -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel  /bin/bash
```

下载torch 1.6版本的官方预测库，并解压，会得到一个libtorch文件夹
```
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu101.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cu101.zip
```

然后克隆此仓库，并将之前的libtorch文件夹放到torch目录下
```
git clone https://github.com/Aurelius84/Inference_benchmark.git
cd torch
cp -r your/path/to/libtorch .
```

inference目录下有一个`image_classification.cpp`，是resnet50/mobileNetv1的预测样例代码，可以编译测试：
```
./re_build.sh
```

执行完毕后，会在当前目录生成一个`image_classification_exe`可执行程序。如果本地有resnet的预测模型，则可以执行预测latency的评测：
```bash
./test_latency ../dy2stat/resnet50.pt

# 注意 torch的.pt模型需要用1.6.0版本的torch保存。
```

### 二、添加新模型测试流程

#### 1. 模型准备
测试paddle和竞品torch、tf的动转静预测性能，首先需要保存 **动转静** 后的模型。

paddle可以通过`@to_static`装饰`forward`函数，然后调用`jit.save`保存。
```python
import paddle
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.jit import to_static


def save_paddle_resnet(model, model_name):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        if '101' in model_name:
            net = model(101, class_dim=1000)
        else:
            net = model(class_dim=1000)
        net.forward = to_static(net.forward, [InputSpec([None, 3, 224, 224], name='img')])
        config = paddle.jit.SaveLoadConfig()
        config.model_filename = 'model'
        config.params_filename = 'params'
        paddle.jit.save(net, model_path=paddle_model_dir + model_name, configs=config)

```

torch可以通过`torch.jit.script`处理model，然后调用`model.save(path)`即可保存`.pt`模型
```python
import torch
import torchvision.models as models

def save_torch_resnet101():
    resnet = models.resnet101(pretrained=False)
    resnet = torch.jit.script(resnet)
    resnet.save(torch_model_dir + "resnet101.pt")
```

目前需要测的模型列表是：resnet50、mobilenetV1、seq2seq、ptb_lm、yolov3

paddle和torch的模型实现，见仓库：https://github.com/phlrain/example


#### 2. 预测接口开发
paddle的预测接口开发，可以参考`paddle/inference/image_classification.cc`中的代码。

可以直接copy一份，修改一下输入的数据shape即可，即修改如下代码行：
```cpp
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);
input_t->Reshape({batch_size, channels, height, width});
input_t->copy_from_cpu(input);
```

torch的预测接口开发更加简单，可以参考`torch/inference/image_classification.cpp`中的代码。

可以直接copy一份，修改一下`std::vector`的inputs中的数据即可：
```cpp
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({batch_size, 3, 224,224}).to(device));
```

#### 3. 编译可执行文件
此步需要修改CMakeLists.txt，在`foreach`循环的文件中，添加新增的文件，会编译生成一个`your_new_file_exe`的可执行文件。
```
set(PredictorSRCFiles "image_classification.cpp"; "your_new_file.cpp")
```

#### 4. 测试预测latency
执行可执行文件，load模型，会输出预测的时间。

