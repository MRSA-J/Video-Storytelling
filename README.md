### Youtube Teller -  a  Video StoryTelling model to extract people's daily activity

### Description
Video Description/Storytelling is a complicated task. For this final project, we are trying to solve a video-caption problem which specifically focuses on people's daily activities.
Our original plan was to focus on cooking activity - caption paring dataset and generate a cooking guide from that dataset. But after we realize that that dataset is really hard to be preprocessed as well as with the previous work related to it is very little, we switch to this [people's daily activities video caption pairing dataset](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/).

### DataSet
[YouTubeClips dataset](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)

Download both the `Microsoft Research Video Description Corpus` and `YouTubeClips.tar` from the above link.

#### Form
The Video is named using its id: For example: `-4wsuPCjDBc_5_15` where `_5_15` means from the 5th second to the 10th second

The Description txt is in the following form: <br>
```
-4wsuPCjDBc_5_15 a squirrel is eating a peanut in it's shell 
-4wsuPCjDBc_5_15 a chipmunk is eating 
-4wsuPCjDBc_5_15 a chipmunk is eating a peanut 
-4wsuPCjDBc_5_15 a chipmunk is eating a nut 
-4wsuPCjDBc_5_15 a squirrel is eating a nut 
-4wsuPCjDBc_5_15 a squirrel is eating a whole peanut 
-4wsuPCjDBc_5_15 a squirrel is eating a peanut 
...
```

With the `-4wsuPCjDBc_5_15` being the video id and `a squirrel is eating a peanut in its shell` being the description. Since the captions all have similar meanings, this task is more like a video caption task instead of a storytelling one.


#### Usage (Dataset Preprocessing)
1. After downloading the dataset, create a folder `data` which is in the same hierarchy with our code and put the unzipped version of `YouTubeClips.tar` and `Microsoft Research Video Description Corpus` inside the data folder.
2. run `python convert_video_to_image.py`
This helps convert our video to sequence of images
3. run `python preprocess.py` to preprocessing
This helps us build our training/testing/validation dataset
4. Set the parameters and train the model like the following
`python train.py --only_prefix --out_dir ./msv_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40`
5. Use the trained model to predict result
`python predict.py`

### Methodology
We choose to treat this task as a sequence-to-sequence generation task.<br> 

Firstly we use the `cv2` package to help us turn video into a sequence of image frames. We set the frameRate to be 1, that means 1s per image frame. We choose this setting as our videos in our dataset are mainly 5s or 10s. We think 1s per image frame is enough for our training. And since our dataset is already very big, we want to make our lives easier by limiting the # of images that the video can be turned into. Feel free to make a larger image frame set corresponding to the video if you have more time. <br>

Then, we use 3 different ways to process our image frames, so that it can be put into the video. Below are the 3 ways of input (and we will justify our choice in our report). We make such decision mainly based on the feature of our dataset:

1. Turn our video into `random single image` frames.
2. Turn our video into `mean image` frames. That means the mean of every image frame corresponding to that video.
3. Turn our video into `sequential image` frames. And we will use positional encoding afterwards in our transformer part to deal with it.

Afterwards, we use the same method described in [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/pdf/2111.09734.pdf), remove the pretrained on COCO dataset's weight and apply our own creativity (positional encoding) on the model. We make this choice, as we believe using prefix training to capture the video/image feature is a very lighted way of training and can achieve good results without the need of managing to "train too much".

Todo: add model structure and explanation

### Metrics
We plan to test our video caption model on the test dataset of uncaptioned videos to generate their captions. We evaluate the performance of our model on the similarity of the generated sentences and standard answers. Specifically, We think the n-gram BLEU score is an appropriate metric to evaluate the accuracy of our captions. The baseline model (Vision Transformer) can achieve 68.4 1-gram BLEU score and 50.7 5-gram BLEU score. We hope to improve the performance in some specific subjects, to achieve higher BLEU scores than the baseline model.

### Our Generation Examples
| Video                         | 1                 |2             | 3               | 4               |
| ----------------------   | ----------- |----------- |----------- |----------- |
| Video ID          |  `HkpUWzNNVt4_20_30`  | `RMznbCn5sQs_0_10` | `aM-RcQj0a7I_37_55`| `R8FDJgVW3Vc_0_4` |
| Example Image Frame |                 |               |                  |                  |   
| Ground Truth Sentence| two couples are interacting.  | zebra are running in an enclosed area.              | chicken is being stirred in boiled.      | a woman is tapping her nails.                 |   
| Single                           | two men are talking on a cell phone. | a tiger is walking. | a man is stirring a pot of water. | someone is peeling a pencil. |   
| Mean                            |  a woman is talking to a man.  | a wild animal is walking on a grassy area.   | a woman is stirring a large pot of water.  | a woman is applying a pencil to a nail.    | 
| Sequential                   |                |               |                  |                   | 

As above, single, mean, sequential means different preprocessing methods and how image frames are selected and passed to the model.

### Contributor
Chen Wei (cwei24), Yuan Zang (yzang6), Yunhao Luo (yluo73)

### Division of labor 
We plan on working equally across X aspects of the project:
1. Preprocess the data: Chen Wei, Yuan Zang
2. Model Architecture
  - Caption Encoder (Use GPT2 pretrained model and add padding, mask): Chen Wei, Yuan Zang
  - Video/image clips encoder: Yuan Zang
  - Multi-head attention Transformer: Yuan Zang, Chen Wei
  - Transformer with Positional Encoding to encode position information: Yuan Zang
  - Evaluation (BLEU, METEOR, CIDEr, SPICE): Yunhao Luo
3. Fine-tuning and Visualization: Yuan Zang
4. Ablation study: Chen Wei
5. Write the report and make the poster: Chen Wei

### Ethics
##### What broader societal issues are relevant to your chosen problem space?
In this project, we aim at generating high-quality, articulated text descriptions given videos as input.  To this end, we can improve the accessibility of various videos in the wild, which hopefully can benefit  users. By the generated descriptions/tags, we can also sort and categorize massive video sets.
##### Why is Deep Learning a good approach to this problem?
Deep learning is currently the most popular and accurate method for computer vision/natural language processing. As for video understanding, by using neural networks with convolutional/attention, the model can learn effective representation. In addition, deep learning methods can achieve end-to-end modeling and are more flexible than traditional methods that usually use handcrafted features (lacking generalizability to  other datasets). 

### Related Work
- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/pdf/2111.09734.pdf)