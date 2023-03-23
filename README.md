# Implementation of paper "Re-thinking Model Inversion Attacks Against Deep Neural Networks" - CVPR 2023

## 1. Setup Environment
This code has been tested with Python 3.7, PyTorch 1.11.0 and Cuda 11.3. 

```
conda create -n MI python=3.7

conda activate MI

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

## 2. Prepare Dataset & Checkpoints

<!-- - CelebA: download and extract the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download-directory). Then, place the `img_align_celeba` folder to `.\datasets\celeba`

- FFHQ: download and extract the [FFHQ](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set). Then, place the `thumbnails128x128` folder to `.\datasets\ffhq` -->

* Down load target models and generator at https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN and https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70

* Extract and place the two folders at `.\datasets` and `.\checkpoints`
  

## 3. Training the classifier (Optinal)

- Modify the configuration in `.\config\celeba\classify.json`
- Then, run the following command line to get the target model
  ```
  python train_classifier.py
  ```
Note that in this repo, we provide training code for training classifiers of KEDMI/GMI experiments on CelebA dataset. Other checkpoints for the three models (i.e., VGG16, IR152, Facenet can be downloaded at https://drive.google.com/drive/folders/14JJxZr2pboHyXwV00kyv9cqxUKKSB8I-?usp=share_link

## 4. Training GAN (Optinal)

SOTA MI attacks work with a general GAN, therefore. However, Inversion-Specific GANs help improve the attack accuracy. In this repo, we provide codes for both training general GAN and Inversion-Specific GAN.

### 4.1. Build a inversion-specific GAN 
* Modify the configuration in
  * `./config/celeba/training_GAN/specific_gan/celeba.json` if training a Inversion-Specific GAN on CelebA
  * `./config/celeba/training_GAN/specific_gan/ffhq.json` if training a Inversion-Specific GAN on FFHQ
  
* Then, run the following command line to get the Inversion-Specific GAN
    ```
    python train_gan.py --configs path/to/config.json --mode "specific"
    ```

### 4.2. Build a general GAN 
* Modify the configuration in
  * `./config/celeba/training_GAN/general_gan/celeba.json` if training a Inversion-Specific GAN on CelebA
  * `./config/celeba/training_GAN/general_gan/ffhq.json` if training a Inversion-Specific GAN on FFHQ
  
* Then, run the following command line to get the General GAN
    ```
    python train_gan.py --configs path/to/config.json --mode "general"
    ```

Pretrained general GAN and Inversion-Specific GAN can be downloaded at https://drive.google.com/drive/folders/1_oyT_JMBym_jse5HcoivFSv4GkpFN5Nz?usp=share_link


## 5. Learn augmented models
We provide code to train augmented models (i.e., `efficientnet_b0`, `efficientnet_b1`, and `efficientnet_b3`) from a ***target model***.
* Modify the configuration in
  * `./config/celeba/training_augmodel/celeba.json` if training an augmented model on CelebA
  * `./config/celeba/training_augmodel/ffhq.json` if training an augmented model on FFHQ
  
* Then, run the following command line to get the General GAN
    ```
    python train_gan.py --configs path/to/config.json
    ```

Pretrained augmented models can be downloaded at https://drive.google.com/drive/folders/12Ib5N9jRkApaVFrUu33S4nexlJwZuCoJ?usp=share_link


## 6. Model Inversion Attack

* Modify the configuration in
  * `./config/celeba/attacking/celeba.json` if training an augmented model on CelebA
  * `./config/celeba/attacking/ffhq.json` if training an augmented model on FFHQ

* Important arguments:
  * `method`: select the method either ***gmi*** or ***kedmi***
  * `variant` select the variant either ***baseline***, ***L_aug***, ***L_logit***, or ***ours***

* Then, run the following command line to attack
    ```
    python recovery.py --configs path/to/config.json
    ```

## 7. Evaluation

After attack, use the same configuration file to run the following command line to get the result:\
```
python evaluation.py --configs path/to/config.json
```




## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.


<a id="2">[1]</a>  Si Chen, Mostafa Kahla, Ruoxi Jia, and Guo-Jun Qi. Knowledge-enriched distributional model inversion attacks. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16178–16187, 2021
