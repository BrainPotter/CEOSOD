# ConsistencySOD: A One-Step, Encoder-Decoder, and Consistency-Model-based Small Object Detector in Aerial Images

## 1. Network Structure
### 1.1 The whold framework of the ConsistencySOD:

<img src="graphs/pic1.png" width="720" height="450"/>

### 1.2 The Consistency Model of the ConsistencySOD:

<img src="graphs/pic2.png" width="720" height="140"/>

## 2. Abstrat

Detecting small objects in aerial images is significantly challenging due to their non-uniform distribution and severe scale variations caused by changing viewing angles. In addition, because of the limited computational power embedded in Unmanned Aerial Vehicles (UAVs), balancing the detection accuracy and efficiency remains a key problem to be addressed. Existing studies, e.g., Feature Pyramid Network (FPN)-based algorithms, concentrate on increasing the resolution of input feature maps. However, features of small objects are easily affected by unpredictable noise from the background. In this work, we tackle these issues by designing a new generative encoder-decoder small object detection (SOD) framework termed ConsistencySOD. ConsistencySOD leverage the self-consistency property provided by the well-known Consistency model, one of the recent advancements of Diffusion models, enabling the ``one-step" inference. First, we reformulate the SOD task as a Noise-to-Box procedure. We then apply the Consistency Model to initialize the diffusion process with Gaussian noisy bounding boxes derived from their corresponding ground-truth annotations. We next introduce a denoising sampling strategy to classify and locate small objects by iterative refining their Gaussian distributions. We finally comprehensively evaluate our proposed framework on several UAV SOD benchmarks, including VisDrone and UAVDT. Experimental results corroborate that ConsistencySOD performs better than the state-of-the-art methods.


## 3. Contributions 

<ul>
    <li>
        <h3></h3>
        <p>Introducing Consistency Models in the field of SOD for the first time. Enhancing the model's inference speed and computational efficiency of SOD by converting Gaussian noisy bounding box candidates into ground-truth ones using only "one-step" process.</p>
    </li>
    <li>
        <h3></h3>
        <p>Designing a new denoising framework that uses a small number of iterations for Gaussian noise addition and removal. Superior SOD performance is achieved by redesigning the the overall loss by considering the loss functions at time steps $t$ and $t+1$</p>
    </li>
</ul>

## 4. Experimental results
<table border="1">
  <tr>
    <th>Method</th>
    <th>AP</th>
    <th>AP_50</th>
    <th>AP_S</th>
    <th>Download</th>
  </tr>
  <tr>
    <td>VisDrone-SwinBase</td>
    <td>32.7</td>
    <td>55.6</td>
    <td>23.6</td>
    <td><a href="https://drive.google.com/file/d/1lH21oidzf2PbP3IgQEBuACpMr7Y-9umy/view?usp=drive_link" download>model</a></td>
  </tr>
  <tr>
    <td>UAVDT-SwinBase</td>
    <td>19.4</td>
    <td>32.1</td>
    <td>16.3</td>
    <td><a href="https://drive.google.com/file/d/1qimluBul5EZyjeNTZDIz7IEPN3n7Gj8o/view?usp=drive_link" download>model</a></td>
  </tr>
</table>

## 5. Environmental initialization
1. Install anaconda, and create conda environment;
<pre>
conda create -n yourname python=3.8
</pre>
2. PyTorch ≥ 1.9.0 and torchvision that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.

3. Install Detectron2
<pre>
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
</pre>

## 6. Preparing data
<pre>
mkdir -p datasets/visdrone
mkdir -p datasets/uavdt
</pre>

You need to download the VisDrone dataset from its [official website](https://aiskyeye.com/)

You need to download the UAVDT dataset from its [official website](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5/)


## 7. Preparing pretrain models
<pre>
mkdir models
cd models
# ResNet-101
wget https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/torchvision-R-101.pkl

# Swin-Base
wget https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/swin_base_patch4_window7_224_22k.pkl
</pre>

## 8. Training
<pre>
python train_visdrone.py --num-gpus 4 \
  --config-file configs/consistencysod.visdrone.swinbase.500boxes.yaml
</pre>

We provide several backbones including ResNet-50, ResNet-101, and Swin-Transformer for training and inference. You can change the backbone by choosing different yaml files in configs folder.

## 9. Evaluating
<pre>
python train_visdrone.py --num-gpus 4 \
  --config-file configs/diffdet.yourdataset.yourbakbone.yaml \
  --eval-only MODEL.WEIGHTS path/to/model.pth
</pre>

## 10. Inference Demo with Pre-trained Models

Inference Demo with Pre-trained Models
We provide a command line tool to run a simple demo following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/demo#detectron2-demo).

<pre>
python demo.py --config-file configs/diffdet.yourdataset.yourbakbone.yaml \
    --input image.jpg --opts MODEL.WEIGHTS path/to/model.pth
</pre>

You need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

## 11. Acknowledgement
A large part of the code is borrowed from DiffusionDet, Consistency models, and ConsistencyDet. Much thanks for their excellent works.
<pre>
@inproceedings{chen2023diffusiondet,
  title={Diffusiondet: Diffusion model for object detection},
  author={Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19830--19843},
  year={2023}
}

@article{song2023consistency,
  title={Consistency models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023}
}

@article{jiang2024consistencydet,
  title={ConsistencyDet: A Robust Object Detector with a Denoising Paradigm of Consistency Model},
  author={Jiang, Lifan and Wang, Zhihui and Wang, Changmiao and Li, Ming and Leng, Jiaxu and Wu, Xindong},
  journal={arXiv preprint arXiv:2404.07773},
  year={2024}
}

</pre>

