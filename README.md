### WeCook - a  Video StoryTelling model to guide cooking

### Team Member
Chen Wei (cwei24), Yuan Zang (yzang6), Yunhao Luo (yluo73)

### Introduction
We are trying to solve a video-caption problem which specifically focuses on a cooking activity - caption paring dataset.

### Related Work
- [MMAC captions dataset introduction paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475557)
- [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/pdf/1904.01766v2.pdf)
- [UniVL: A Unified Video and Language Pre-Training Model for
Multimodal Understanding and Generation]
(https://arxiv.org/pdf/2002.06353v3.pdf)

### Data
[MMAC captions dataset ](https://github.com/hitachi-rd-cv/mmac_captions)  <br>
MMAC dataset is a dataset for sensor-augmented egocentric-video captioning. This dataset contains 5,002 activity descriptions. <br>

Samples: <br>
![](https://raw.githubusercontent.com/hitachi-rd-cv/mmac_captions/main/.github/MMAC_Captions.jpg)

- Spreading tomato sauce on pizza crust with a spoon.
- Taking a fork, knife, and peeler from a drawer.
- Cutting a zucchini in half with a kitchen knife.
- Moving a paper plate slightly to the left.
- Stirring brownie batter with a fork.

##### Training/Validation/Test split
- Training: 2923
- Validation: 838
- Test:1241

The given dataset is in tsv form:

|record_no |video_id |vid|start_time|end_time|start_frame|end_frame|caption|
|  ----  | ----  | ----| ----|----|----|----|----|
| 0 |S50_Brownie_7150991-203| f141_i023_a00|151.0|152.0|4530|4590|taking a kitchen knife from a drawer .|
| 1 |S47_Salad_7150991-196| f130_i028_a00|214.0|246.0|6420|7410| peeling a zucchini with a peeler .|
| 2 |S22_Salad_7150991-952| f60_i000_a00|6.0|9.0|180|300|taking a bowl from a cabinet . |

We will only train on one category for simplicity and time concern. We chose subcategory Pizza.

### Methodology
Firstly, we plan to train a video encoder, which can consist of a CNN-based image encoder and a Transformer-based encoder to encode the temporal information of the video. Then we can utilize the video encoding as the input of a language model, which can be a LSTM or a Transformer-decoder, to generate the captions. We might consider using the word embeddings from [GloVe dataset](https://nlp.stanford.edu/projects/glove/).
 
We will train our model on the training set using cross entropy loss combined with some custom loss. We will then validate the model on validation set to further tune the model.

### Metrics
We plan to test our story-telling model on the test dataset of uncaptioned videos to generate their captions. We evaluate the performance of our model on the similarity of the generated sentences and standard answers. Specifically, We think the n-gram BLEU score is an appropriate metric to evaluate the accuracy of our captions. The baseline model (Vision Transformer) can achieve 68.4 1-gram BLEU score and 50.7 5-gram BLEU score. We hope to improve the performance in some specific subjects, to achieve higher BLEU scores than the baseline model.

### Ethics
##### What broader societal issues are relevant to your chosen problem space?
In this project, we aim at generating high-quality, articulated text descriptions given videos as input.  To this end, we can improve the accessibility of various videos in the wild, which hopefully can benefit  users. By the generated descriptions/tags, we can also sort and categorize massive video sets.
##### Why is Deep Learning a good approach to this problem?
Deep learning is currently the most popular and accurate method for computer vision/natural language processing. As for video understanding, by using neural networks with convolutional/attention, the model can learn effective representation. In addition, deep learning methods can achieve end-to-end modeling and are more flexible than traditional methods that usually use handcrafted features (lacking generalizability to  other datasets). 


### Division of labor
- ...
- ...
