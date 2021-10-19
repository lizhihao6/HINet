``` bash
# install dvs-deblur
git clone https://git-core.megvii-inc.com/lizhihao/dvs-deblur.git
cd HINet
pip install -r requirements.txt
export CUDA_HOME=XXX
python setup.py develop
# download experiments/pretrained_models/DVS_HINet-GoPro.pth model 
# from https://box.nju.edu.cn/f/4accea732b5e418193e8/
# or
oss cp s3://lzh-share/models/DVS_HINet-GoPro.pth ./experiments/pretrained_models/DVS_HINet-GoPro.pth

# install v2e
cd ~ && git clone https://github.com/lizhihao6/v2e.git
ln -s ~/v2e .v2e
cd .v2e && pip install -r requirements.txt && python setup.py develop && cd ../
# download .v2e/.pretrain/SuperSloMo39.ckpt 
# from https://drive.google.com/u/0/uc?id=17QSN207h05S_b2ndXjLrqPbBTnYIl0Vb&export=download
# or
oss cp s3://lzh-share/models/SuperSloMo39.ckpt ./.v2e/.pretrain/SuperSloMo39.ckpt
!!! remember to test v2e before generate events
# test code
# download .v2e/input/tennis.mov from https://drive.google.com/file/d/1dNUXJGlpEM51UVYH4-ZInN9pf0bHGgT_/view

# use rlanuch multigpus
echo "alias rr='rlaunch --cpu=48 --gpu=8 --memory=169152 --replica-restart=on-failure -P 1 --preemptible=no '" >> ~/.zshrc
```

---


<details>
  <summary>Image Deblur - GoPro dataset (Click to expand) </summary>

* **generate datasets **(!!! only need when you want to regenerate events !!!)**
    * ```python scripts/data_preparation/gopro.py```

* prepare datasets
  ``` bash
  oss sync s3://lzh-share/GOPRO_Large/ /data/GOPRO_Large/
  oss sync s3://lzh-share/GoPro/ /data/GoPro/
  ln -s /data/GoPro datasets/GoPro
  ln -s /data/GOPRO_Large datasets/GOPRO_Large 
  ```

* eval
    * download [pretrained model](https://drive.google.com/file/d/1dw8PKVkLfISzNtUu3gqGh83NBO83ZQ5n/view?usp=sharing) to
      ./experiments/pretrained_models/HINet-GoPro.pth
    * ```python basicsr/test.py -opt options/test/GoPro/HINet-GoPro.yml  ```

* train

    * ```python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro/HINet.yml --launcher pytorch```

</details>


<details>
  <summary>Image Deblur - stereo dataset (Click to expand) </summary>

* **generate datasets **(!!! only need when you want to regenerate events !!!)**
  ```
  # x16 VFI
  git clone https://github.com/lizhihao6/ABME.git
  python3 stereo_blur_x16.py [0-8] (use rlanch) 
  # generate events
  python3 scripts/data_preparation/dvs_genertor.py [0-8] (use rlanuch)
  # crop
  cd scripts/data_preparation && python3 stereo.py (use rlanuch)
  # make nori
  cd scripts/data_preparation/make_nori_ll3/make_stereo_nori.py (use rlanuch)
  ```

* prepare datasets
  ``` bash
  # prepare training dataset
  oss cp s3://lzh-share/stereo_blur_data/train_v4.nori.json /data/stereo_blur_data/train_v4.nori.json
  oss cp s3://lzh-share/stereo_blur_data/test_v4.nori.json /data/stereo_blur_data/test_v4.nori.json
  ln -s /data/stereo_blur_data datasets/stereo_blur_data
  # prepare test dataset
  oss sync s3://lzh-share/MiDVS /data/MiDVS
  ```

* eval
    * oss cp /lzh/share/models/net_g_40000.pth ./experiments/Stereo-DVS-HINet/models/net_g_40000.pth
    * ```python basicsr/test.py -opt options/test/Stereo/HINet-MiDVS.yml  ```

* train
    * oss cp /lzh/share/models/HINet-REDS.pth ./experiments/pretrained_models/HINet-REDS.pth
    * ``` python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/Stereo/DVS_HINet.yml --launcher pytorch```

</details>
