### About Life -  a  Video StoryTelling model to extract people's daily activity

### Description
Video Description/Storytelling is a complicated task. For this final project, we are trying to solve a video-caption problem which specifically focuses on people's daily activities.
Our original plan was to focus on cooking activity - caption paring dataset and generate a cooking guide from that dataset. But after we realize that that dataset is really hard to be preprocessed as well as with the previous work related to it is very little, we switch to this [people's daily activities video caption pairing dataset](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/).

### DataSet
[YouTubeClips dataset](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)

Download both the `Microsoft Research Video Description Corpus` and `YouTubeClips.tar` from the above link.

#### Form
The Video is named using its id: For example: `-4wsuPCjDBc_5_15` where `_5_15` means from the 5th second to the 10th second

The Description txt is in the following form:
-4wsuPCjDBc_5_15 a squirrel is eating a peanut in it's shell <br>
-4wsuPCjDBc_5_15 a chipmunk is eating <br>
-4wsuPCjDBc_5_15 a chipmunk is eating a peanut <br>
-4wsuPCjDBc_5_15 a chipmunk is eating a nut <br>
-4wsuPCjDBc_5_15 a squirrel is eating a nut <br>
-4wsuPCjDBc_5_15 a squirrel is eating a whole peanut <br>
-4wsuPCjDBc_5_15 a squirrel is eating a peanut <br>
...

With the `-4wsuPCjDBc_5_15` being the video id and `a squirrel is eating a peanut in its shell` being the description. Since the captions all have similar meanings, this task is more like a video caption task instead of a storytelling one.


#### Usage (Dataset Preprocessing)
1. After downloading the dataset, create a folder `data` which is in the same hierarchy with our code and put the unzipped version of `YouTubeClips.tar` and `Microsoft Research Video Description Corpus` inside the data folder.
2. run `python convert_video_to_image.py`
This helps convert our video to sequence of images
3. run `python preprocess.py` to preprocessing
This helps us build our training/testing/validation dataset

### Methodology
We choose to treat this task as a sequence-to-sequence generation task. We ...


Firstly, we plan to train a video encoder, which can consist of a CNN-based image encoder and a Transformer-based encoder to encode the temporal information of the video. Then we can utilize the video encoding as the input of a language model, which can be a LSTM or a Transformer-decoder, to generate the captions. We might consider using the word embeddings from [GloVe dataset](https://nlp.stanford.edu/projects/glove/).
 
We will train our model on the training set using cross entropy loss combined with some custom loss. We will then validate the model on validation set to further tune the model.

### Metrics
We plan to test our story-telling model on the test dataset of uncaptioned videos to generate their captions. We evaluate the performance of our model on the similarity of the generated sentences and standard answers. Specifically, We think the n-gram BLEU score is an appropriate metric to evaluate the accuracy of our captions. The baseline model (Vision Transformer) can achieve 68.4 1-gram BLEU score and 50.7 5-gram BLEU score. We hope to improve the performance in some specific subjects, to achieve higher BLEU scores than the baseline model.

### Contributor
Chen Wei (cwei24), Yuan Zang (yzang6), Yunhao Luo (yluo73)

### Division of labor 
We plan on working equally across X aspects of the project:
1. Preprocess the data: Chen Wei, Yuan Zang
2. Model Architecture - 2 encoders below form a video encoder: 
  - a 3d-CNN-based image encoder (to get C3D features): Chen Wei
  - a Transformer-based encoder: Yunhao Luo
  - LSTM or Transformer-decoder: Yuan Zang
3. Fine-tuning, Evaluation and Visualization: Together
4. Ablation study (maybe)
5. Write the report and make the poster: Together

### Ethics
##### What broader societal issues are relevant to your chosen problem space?
In this project, we aim at generating high-quality, articulated text descriptions given videos as input.  To this end, we can improve the accessibility of various videos in the wild, which hopefully can benefit  users. By the generated descriptions/tags, we can also sort and categorize massive video sets.
##### Why is Deep Learning a good approach to this problem?
Deep learning is currently the most popular and accurate method for computer vision/natural language processing. As for video understanding, by using neural networks with convolutional/attention, the model can learn effective representation. In addition, deep learning methods can achieve end-to-end modeling and are more flexible than traditional methods that usually use handcrafted features (lacking generalizability to  other datasets). 

### Related Work
- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/pdf/2111.09734.pdf)
