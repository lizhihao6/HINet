```
# install dvs-deblur
git clone https://git-core.megvii-inc.com/lizhihao/dvs-deblur.git
cd HINet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
# download experiments/pretrained_models/DVS_HINet-GoPro.pth model 
# from https://box.nju.edu.cn/f/4accea732b5e418193e8/
# or
oss cp s3://lzh-share/models/DVS_HINet-GoPro.pth ./experiments/pretrained_models/DVS_HINet-GoPro.pth

# install v2e
cd ~ && git clone https://github.com/lizhihao6/v2e.git
ln -s ~/v2e .v2e
# download .v2e/.pretrain/SuperSloMo39.ckpt 
# from https://drive.google.com/u/0/uc?id=17QSN207h05S_b2ndXjLrqPbBTnYIl0Vb&export=download
# or
oss cp s3://lzh-share/models/SuperSloMo39.ckpt ./.v2e/.pretrain/SuperSloMo39.ckpt

# use rlanuch multigpus
echo "alias rr='rlaunch --cpu=48 --gpu=8 --memory=169152 --replica-restart=on-failure -P 1 --preemptible=no '" >> ~/.zshrc
```
---

<details>
  <summary>Image Deblur - GoPro dataset (Click to expand) </summary>

* prepare datasets
  ```
  oss sync s3://lzh-share/GOPRO_Large/ /data/GOPRO_Large/
  oss sync s3://lzh-share/GoPro/ /data/GoPro/
  ln -s /data/GoPro datasets/GoPro
  ln -s /data/GOPRO_Large datasets/GOPRO_Large 
  ```
* generate events
  ```
  python scripts/data_preparation/generate_events.py
  python scripts/data_preparation/gopro.py

* eval
  * download [pretrained model](https://drive.google.com/file/d/1dw8PKVkLfISzNtUu3gqGh83NBO83ZQ5n/view?usp=sharing) to ./experiments/pretrained_models/HINet-GoPro.pth
  * ```python basicsr/test.py -opt options/test/GoPro/HINet-GoPro.yml  ```
  
* train

  * ``` python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/HINet.yml --launcher pytorch```

</details>

