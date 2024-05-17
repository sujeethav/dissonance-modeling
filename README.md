# dissonance-modeling

This Repository hhas the code for Training Classification Model for Dissonance Detection. We are training RoBERTa model for downstream Dissonance classification task.

## Files

1. [run.py](./src/run.py): This is a driver code for Model Training. This also has setup for Hyperparemeter tuning with Optuna.
2. [model.py](./src/model.py): This has class objects for Different Model Architectures Experiemented. The below are he different Architectures Tried.
   1. DissonanceClassifierTweetandDus: This model implements tweet level batching. This is a Tweet[SEP]DU1[SEP]DU2] Architecture.
   2. BeforAfterWholeDissonance: This is a Tweet[SEP]DU1[SEP]DU2[SEP]Before DU1[SEP]After DU2 Architecture.
   3. DissonanceClassifierCrossAttn: This is a Cross Attention Model which is contextualising within tweet du1[SEP]du2[SEP]du3[SEP]du4[SEP]....
   4. DissonanceClassifierFullTweetSep: This architecure is contextualising within tweet du1[SEP]du2[SEP]du3[SEP]du4[SEP].... but without the cross attention layer.
This file also has multiple other model trials experimented in the project.
 
4. [dataset.py](./src/dataset.py): This has classes for createng different dataset for its respective model architecture mentioned above. The specific classes does the necessarcy data preprocessing for the respective models. This is more of a data preparation tasks for each model. Models where we are contextualising with tweets and  withing tweets needed some preparation steps.

## Evaluation Results

The results of these Models are available on my WandB:

1. [DissonanceClassifierTweetandDus](https://api.wandb.ai/links/sujeethav/t6w69id3)
2. [BeforAfterWholeDissonance](https://api.wandb.ai/links/sujeethav/ix2pvwli)
3. [DissonanceClassifierCrossAttn](https://api.wandb.ai/links/sujeethav/45da2r2k)
4. [DissonanceClassifierFullTweetSep](https://api.wandb.ai/links/sujeethav/oa96pmuz)

The DissonanceClassifierFullTweetSep And DissonanceClassifierCrossAttn Models were tested on Debate and PDTB dataset as well. The below are the WandB reports for it:

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
#### Command line Arguements
- `devfile`: Pass the path of the Dev or Eval file for Model Evaluation
- `testfile`: Pass the Test File for model evaluation at the end of Model Training.
- `name`: This the name you can provide for the trial.
- `arc`: Mention the name of the architecture you want to train. The following are the choices ``"basic", "before_after", "before_after_sep","tweet_sep","cross_attn", "cross_attn_tweet","kialo_cross_attn", "kialo_sep", 'before_after_whole','tweet_batching', 'pdtb_sep','pdtb_cross_attn'``
