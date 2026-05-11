# Multimodal Emotion Recognition 

## 1. Task Definition

Design 3 deep learning models - CNN, RNN, Fusion - that classify human emotions using both audio data and generated text transcripts.

- 8 emotion classes - neutral, calm, happy, sad, angry, fearful, disgust, surprise.


Male and female voices differ in pitch range - female voices (180–300Hz, male 85–180Hz). A Mel-Spectrogram sees these as different energy distributions across frequency bins. Training a single 8-class model means it has to learn both the pitch shift and the emotion patterns. Separating by gender into 16 classes would make it easier for the CNN to learn emotion patterns. However, with only 1440 clips across 16 classes that leaves roughly 90 samples per class, and with an 80/10/10 split the test set would have around 9 samples per class - not enough for a reliable evaluation. Given this tradeoff, I kept 8 emotion classes and injected gender as an additional input feature via an embedding layer, so the model is aware of gender.

---

## 2. Dataset

**RAVDESS Emotional Speech Audio**
- 1440 audio files, 16-bit 48kHz WAV
- 24 actors (12 male, 12 female), 60 clips per actor
- 2 sentences: "Kids are talking by the door" and "Dogs are sitting by the door"

**File naming convention** (dash-separated identifiers):

| Position | Identifier | Values used |
|---|---|---|
| 3 | Emotion | 01–08 -> 8 classes |
| 7 | Actor | 01–24, odd = male, even = female |


- `neutral` has only 96 clips vs 192 for every other emotion - no strong intensity variant for neutral. This imbalance is handled via class weights in the loss function.
- Only two sentences spoken across all 8 emotions - the text branch carries near-zero emotion signal by construction, not due to any model limitation.
- ~180 clips per emotion total. After an 80/10/10 split, the model trains on ~144 clips per class.


---

## 3. Audio Preprocessing

 - convert raw waveforms into fixed-size 2D tensors suitable for CNN input.

- A raw waveform would contain (`sampling_rate` * `time`) amplitude samples, in this case around 150k samples per clip, if we feed it directly to a CNN it will be very expensive and wont use the time-frequency structure of speech.
- A spectrogram would split the whole duration into smaller segments -> apply fourier transform in each segment to get the frequencies contained in that segment and combine them all to show Frequency vs Time and different colors for Amplitude of each frequency -> Convert to mel scale and decibels we get Mel Spectrogram
[Reference](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505/)
**Pipeline:**

1. Resample to 22050 Hz (librosa standard)
2. Trim silence - each clip has a quiet period before the actor speaks. `top_db=60` would remove anything 60 dB quieter than the loudest moment. It wont touch quiet speech or breath sounds which can carry emotion cues. (tried using top_db=30 which was too aggressive and created a black padding region of nothing)
3. Pad/crop to 3 seconds - A CNN needs identically sized inputs/ After trimming clips vary slightly in length which was enough to cover all the clips
4. Mel Spectrogram 
   - (default value)`n_fft=2048` : window length, gives freq resolution of `22050/2048 ~ 10Hz`
   - `n_mels=64`  : compress the FFT frequency binds to 64 mel-scale bins to reduce dimensionality and preserve important info. 
5. Convert to decibel - Using `ref=1.0` (a fixed reference) while conversion to decibels to preserve absolute energy differences across clips - using `max` would destroy the energy difference
6. Standardise per clip - 0 mean , 1 std. Puts every spectrogram in consistent range to eliminate the effect of volume differences and focus on actual patterns.

* total frames = LEN/HOP_LEN ~ 130 here
Output shape: `(1, 64, 130)` - 1 channel, 64 mel bins, ~130 time frames.

---

## 4. Train / Val / Test Split

**80 / 10 / 10 stratified split.**

- Random splits risk leaving some emotion classes less represented in validation/test. Stratify ensures each split seems same class proportions
- With 1440 samples, 10% test ~ 144 clips and 80% for training since we need to give more importance to training given its small size\

- Class weights : loss for classes with fewer data points is scaled up proportionally
- Formula - $\frac{\text{tot samples}}{\text{n classes * count per class}}$
- So neutral with fewer clips gets penalised more heavily when misclassified

---

## 5. Audio CNN

### 5.1 Architecture

The spectrogram is a `(1,64,130)` tensor - 64 mel freq bins, 130 time frames, 1 channel. A 2D CNN  because emotion signal lives in both axes (pitch and how it changes over time).

1. A square $(n,n)$ kernel would work but it comes with a problem that frequency and time mean completely different things, so a sqaure kernel would have to learn to treat them differently to figure out the low level patterns in each of them before it could start forming combinations of them in later layers. Keeping this in mind I started with:
   1. Frequency scan : `Conv2d(1->16, (5x1)) -> Conv2d(16->32, (5x1))` - To learn which frequency bins activate together at any given time frame. Two layers of this give a combined receptive field of $(9,1)$ while keeping non-linearity than a single $(9,1)$ layer. I kept channels small $1 \rightarrow 16 \rightarrow 32$
   2. Time scan : `Conv2d(32->32, (1x9)) -> Conv2d(32->64, (1x9))` To learn how each frequency band evolves with time. The combined receptive field is $(17,1)$ with same logic as before. Also this block receives frequency features from block 1 so it starts learning combinations already.
      * The padding for each would be $p = \frac{k-1}{2}$ so 2 and 4 respectively
    After both axes have been processed I use $(n,n)$ kernel to learn combinations of already extracted features from them

2. Pooling after 2nd block for dimensionality reduction before convolutions with square kernels. But I didnt want to reduce frequency resolution as it matters more for separating emotions so I used `MaxPool((2,4))` - time 4x and frequency 2x, So shape becomes `(64,32,32)`
3. Next before considering the combinations of patterns I included the gender feature in as it should also be considered so as to introduce the capability of "gender + emotion" features instead of hard "emotion" features only.               
   However each member of the batch is now represented across 64 channels of height 32 and width 27 and I wanted to do it without adding in more dimensions so after searching for such methods I used [Feature independent Linear Modulation (FiLM)](https://ml-retrospectives.github.io/neurips2019/accepted_retrospectives/2019/film/) which applies a linear transformation to the gender embedding
and adds by broadcsting to (H,W) as 

    $\text{feat} = \gamma \cdot \text{feat} + \beta$

This updates the existing feature map without changing its shape

`gender_emb`: `Embedding(2, 8)` - maps gender ID to an 8-dim learned vector.  
`gender_film`: `Linear(8, 128)` - projects to 64 gammas + 64 betas via `.chunk(2, dim=1)`.

4. Cross combinations `Conv2d(64->64, (3x3)) -> Conv2d(64->128, (3x3)) -> MaxPool(2,2) -> Conv2d(128->128, (3x3))`
Now frequency, time and gender components are all present so we can use $(3,3)$ kernels for theri combinations
* Before flattening and completely disregarding temporal / frequency cues used `AdaptiveAvgPool2d((2,2))` to shrink it to (2,2) as a rough version of it then flattening would give 512 values (128* 2 *2).
   
5. Bottleneck `Linear(512->128) -> BatchNormalise -> ReLU -> Dropout(0.5) -> Linear(128->64) -> ReLU -> Linear(64->8)`
BatchNormalisation and Dropout are for regularisation, doing normalisatoin after relu doesnt help much as its already half-zeroed. Dropout is at 0.5 cause with the low numbers of training points risk of overfitting is high.

Finally 64 dim ->8 classes so  8:1 ratio of representation is present. 

| Layer | Output Shape | Params |
|---|---|---|
| **AudioCNN** | [1, 8] | - |
| **frequency_block** | [1, 32, 64, 130] | - |
| Conv2d (1$\rightarrow$16, 5×1) | [1, 16, 64, 130] | 96 |
| BatchNorm2d (16) | [1, 16, 64, 130] | 32 |
| ReLU | [1, 16, 64, 130] | - |
| Conv2d (16$\rightarrow$32, 5×1) | [1, 32, 64, 130] | 2,592 |
| BatchNorm2d (32) | [1, 32, 64, 130] | 64 |
| ReLU | [1, 32, 64, 130] | - |
| **time_block** | [1, 64, 32, 32] | - |
| Conv2d (32$\rightarrow$32, 1×9) | [1, 32, 64, 130] | 9,248 |
| BatchNorm2d (32) | [1, 32, 64, 130] | 64 |
| ReLU | [1, 32, 64, 130] | - |
| Conv2d (32$\rightarrow$64, 1×9) | [1, 64, 64, 130] | 18,496 |
| BatchNorm2d (64) | [1, 64, 64, 130] | 128 |
| ReLU | [1, 64, 64, 130] | - |
| MaxPool2d (2×4) | [1, 64, 32, 32] | - |
| **gender_emb** Embedding (2$\rightarrow$8) | [1, 8] | 16 |
| **gender_film** Linear (8$\rightarrow$128) | [1, 128] | 1,152 |
| **cross_block** | [1, 128, 2, 2] | - |
| Conv2d (64$\rightarrow$64, 3×3) | [1, 64, 32, 32] | 36,928 |
| BatchNorm2d (64) | [1, 64, 32, 32] | 128 |
| ReLU | [1, 64, 32, 32] | - |
| Conv2d (64$\rightarrow$128, 3×3) | [1, 128, 32, 32] | 73,856 |
| BatchNorm2d (128) | [1, 128, 32, 32] | 256 |
| ReLU | [1, 128, 32, 32] | - |
| MaxPool2d (2×2) | [1, 128, 16, 16] | - |
| Conv2d (128$\rightarrow$128, 3×3) | [1, 128, 16, 16] | 147,584 |
| BatchNorm2d (128) | [1, 128, 16, 16] | 256 |
| ReLU | [1, 128, 16, 16] | - |
| AdaptiveAvgPool2d (2×2) | [1, 128, 2, 2] | - |
| **bottleneck** | [1, 64] | - |
| Flatten | [1, 512] | - |
| Linear (512$\rightarrow$128) | [1, 128] | 65,664 |
| BatchNorm1d (128) | [1, 128] | 256 |
| ReLU | [1, 128] | - |
| Dropout (0.5) | [1, 128] | - |
| Linear (128$\rightarrow$64) | [1, 64] | 8,256 |
| ReLU | [1, 64] | - |
| **classifier** Linear (64$\rightarrow$8) | [1, 8] | 520 |
| | | |
| **Total params** | | **365,592** |



### 5.2 Training

- Optimiser: Adam, `lr=5e-4`, `weight_decay=1e-4` (the FiLM reference states requirement for careful regulrisation)
- Loss: weighted `CrossEntropyLoss`
- Epochs: 100, batch size 32
- Best checkpoint saved by validation loss

---
## 6. Text GRU

### 6.1 Limitations
* RAVDESS only contains two sentences spoken across all 8 emotions. The text containts zero emtion context and Whisper will transcribe almost identically regardless of emotion. 
* This branch will perform close to random and no choices can change that as its a data problem. So I didnt put much thought into architecture design compared to CNN.

### 6.2 Transcription - Whisper

Whisper is an encoder-decoder transformer trained on 680k hours of speech. The encoder processes the mel spectrogram through self-attention layers to produce context-aware frame representations. The decoder generates the transcript - at each step it attends to the encoder output via cross-attention and to previously generated tokens via masked self-attention.

The `tiny` variant (39M parameters) is used.

### 6.3 Tokenisation

Since only two sentences are there, the vocabulary is tiny and a custom dictionary should work fine. 
- We want to convert each transcript stirng into a sequence of integers that the Embedding layer can look up
  - Needs to be built from train data only
  - Lower case everything and remove punctuations
  - Split on whitespace - each word becomes a token which is fine cause there arent that many words
  - Assign each token a unique integer - add the special tokens for `<PAD>=0` for  fixed length requirement and `<UNK>=1` in case new words which show up during val/test.
- Encode with the created dictionary and pad to `MAX_LEN=15`, shorter sequences are padded with 0's and longer truncated from right.

### 6.4 Architecture - Embedding + GRU

1. Embedding Layer: `nn.Embedding(input_dim,emb_dim,padding_idx=0)`
An embedding lookup table of shape `(VOCAB_SIZE,32)` here where each token is now represented bya 32 dimensional vector.
* `<PAD>` corresponding vector will just be a zero vector
* `<UNK>` corresponding vector wont receive any updates while training cause all tokens in training are known and remain its initialised value and if by chance a new word comes during testing it will just feed random noise to GRU, however thats fine here since RAVDESS has only two sentences so a new word occuring would be extremely rare. It can be handled in case of many rares by changing its value to the mean of other already trained embeddings 
1. GRU: `nn.GRU(32, 64, batch_first=True)`

Process the sequence of now 64-dim embeddings one at a time. At each time step t it computes:
- Update gate value - how much of the previous hidden state to carry forward
- Reset gate value - how much of the prev hidden state is used to compute the next hidden state candidate
The final stats summarises the entire sequence and we use that direclty 
* GRU expects inputs as `(seq_len,batch,inp_dim)` but the embedding outputs as `(batch,seq_len,emb_dim)` so we need `batch_first=True` to match dimensions. 

After linear fc layer with output dim 64 with ReLU activation which is the final bottleneck before classifier.

**Classifier: `Linear(64->64) -> ReLU -> Linear(64->8)`**

### 6.5 Training

- Optimiser: Adam, `lr=5e-4`, `weight_decay=1e-4`
- Loss: weighted `CrossEntropyLoss`
- Epochs: 100, batch size 32

---

## 7. Late Fusion

Both models are trained independently. At inference each produces a softmax probability vector over 8 classes. Three combinations:

**Average:** $P_{fusion} =0.5 \cdot P_{audio} +0.5 \cdot P_{text}$

**Weighted average:** $P_{fusion} =0.7 \cdot P_{audio}+0.3 \cdot P_{text}$

Assigns more trust to the audio model given its known superiority on this dataset.

**Maximum confidence:** for each sample, use the prediction from whichever model has the higher max probability.

Since the text branch performs near-randomly, fusion is expected to either match or slightly degrade audio-only performance 

With the text considerations in mind I only did Late Fusion as Early Fusion didnt make sense in this scenario. I still separated the forward chain bottleneck and classifier so if fusion is required elsewhere I can just use the bottleneck from both models and pass them through a final combined classifier block.

---

## 8. Results

### Accuracy and F1

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Audio CNN | 0.7847 | 0.7936  |
| Text GRU | 0.1389 | 0.0193 |
| Fusion (avg) | 0.7847 | 0.7936 |


### Loss Curves

![img](<loss_curves.png>)

### Confusion Matrix
![img](<conf_mat.png>)

### Per-class Precision, Recall, F1

| Emotion | CNN P | Fusion(avg) P | Fusion(wt) P | GRU P | CNN R | Fusion(avg) R | Fusion(wt) R | GRU R | CNN F1 | Fusion(avg) F1 | Fusion(wt) F1 | GRU F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| angry | 0.813 | 0.813 | 0.813 | 0.000 | 0.684 | 0.684 | 0.684 | 0.000 | 0.743 | 0.743 | 0.743 | 0.000 |
| calm | 0.889 | 0.889 | 0.889 | 0.000 | 0.842 | 0.842 | 0.842 | 0.000 | 0.865 | 0.865 | 0.865 | 0.000 |
| disgust | 0.696 | 0.696 | 0.696 | 0.000 | 0.842 | 0.842 | 0.842 | 0.000 | 0.762 | 0.762 | 0.762 | 0.000 |
| fearful | 0.714 | 0.714 | 0.714 | 0.000 | 0.789 | 0.789 | 0.789 | 0.000 | 0.750 | 0.750 | 0.750 | 0.000 |
| happy | 0.933 | 0.933 | 0.933 | 0.000 | 0.737 | 0.737 | 0.737 | 0.000 | 0.824 | 0.824 | 0.824 | 0.000 |
| neutral | 0.667 | 0.667 | 0.667 | 0.000 | 0.667 | 0.667 | 0.667 | 0.000 | 0.667 | 0.667 | 0.667 | 0.000 |
| sad | 0.714 | 0.714 | 0.714 | 0.000 | 0.750 | 0.750 | 0.750 | 0.000 | 0.732 | 0.732 | 0.732 | 0.000 |
| surprise | 0.857 | 0.857 | 0.857 | 0.139 | 0.900 | 0.900 | 0.900 | 1.000 | 0.878 | 0.878 | 0.878 | 0.244 |

**Inferences:**

- GRU predicts only one emotion for all as expected it doesnt learn any structure related to emotion from text
- Because of above reason fusion model is the same as CNN
- In CNN , neutral and sad are now the most confused (f1 score - 0.667,0.732) of which neutral was expected before from its less clips and apart from that its ambiguity in features.
  
---

## 9. Leaving One Actor Out Evaluation

Leave-one-actor-out: actor 24 is held out completely as the test set, the remaining 23 actors form the train/val pool. The model is evaluated on a speaker it has never heard during training.

Accuracy (actor 24) : 0.5667
 
---
## 10. Augmented Data Evaluation

| Clean | Augmented | Drop |
|-------|-----------|------|
| 0.7847| 0.5903 | 0.1944|

With adding minor perturbations like mild Gaussian noise  and shifting pitch, the model shows considerable drop even for very small noise which is expected because the model was trained on clean data and even shifting pitch a bit causes those bins to be activated which earlier never were causing confusion.
