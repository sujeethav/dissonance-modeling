# dissonance-modeling

This repository contains the code for training a classification model for dissonance detection. We are training a RoBERTa model for the downstream task of dissonance classification.

## Files

1. [run.py](./src/run.py): This is the driver code for model training. It also includes setup for hyperparameter tuning with Optuna.
2. [model.py](./src/model.py): This file contains class objects for the different model architectures experimented with. The architectures include:
   1. DissonanceClassifierTweetandDus: Implements tweet-level batching with a Tweet[SEP]DU1[SEP]DU2 architecture.
   2. BeforeAfterWholeDissonance: Implements a Tweet[SEP]DU1[SEP]DU2[SEP]Before DU1[SEP]After DU2 architecture.
   3. DissonanceClassifierCrossAttn: A cross-attention model that contextualizes within tweet DU1[SEP]DU2[SEP]DU3[SEP]DU4[SEP]...
   4. DissonanceClassifierFullTweetSep: Similar to the previous model but without the cross-attention layer.
   
   This file also includes multiple other model trials experimented with in the project.
 
3. [dataset.py](./src/dataset.py): This file contains classes for creating different datasets for the respective model architectures mentioned above. The specific classes perform the necessary data preprocessing for each model. Models that contextualize with tweets and within tweets require specific preparation steps.

## Evaluation Results

The results of these models are available on my WandB:

1. [DissonanceClassifierTweetandDus](https://api.wandb.ai/links/sujeethav/t6w69id3)
2. [BeforeAfterWholeDissonance](https://api.wandb.ai/links/sujeethav/ix2pvwli)
3. [DissonanceClassifierCrossAttn](https://api.wandb.ai/links/sujeethav/45da2r2k)
4. [DissonanceClassifierFullTweetSep](https://api.wandb.ai/links/sujeethav/oa96pmuz)

The DissonanceClassifierFullTweetSep and DissonanceClassifierCrossAttn models were also tested on the Debate and PDTB datasets. The WandB reports for these tests are available below:

### Debate Results
1. [DissonanceClassifierFullTweetSep](https://api.wandb.ai/links/sujeethav/tutxth2a)
2. [DissonanceClassifierCrossAttn](https://api.wandb.ai/links/sujeethav/4dmuip8w)

### PDTB Results
1. [DissonanceClassifierFullTweetSep](https://api.wandb.ai/links/sujeethav/36o2sa63)
2. [DissonanceClassifierCrossAttn](https://api.wandb.ai/links/sujeethav/5zg4omo1)

## Usage

```bash
python run.py -trainfile train_data.json -devfile dev_data.json -testfile test_data.json --name tweet_sep_model_only_pairs --arc tweet_sep
```

#### Command Line Arguments
- `devfile`: Path to the development or evaluation file for model evaluation.
- `testfile`: Path to the test file for model evaluation at the end of model training.
- `name`: The name you can provide for the trial.
- `arc`: The name of the architecture you want to train. The following are the choices: `"basic", "before_after", "before_after_sep", "tweet_sep", "cross_attn", "cross_attn_tweet", "kialo_cross_attn", "kialo_sep", "before_after_whole", "tweet_batching", "pdtb_sep", "pdtb_cross_attn"`.
