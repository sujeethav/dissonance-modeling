import torch
from model import DissonanceClassifier
from dataset import DissonanceDataset
from transformers import Trainer, TrainingArguments, AdamW, EarlyStoppingCallback
import numpy as np
import random
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Subset
import wandb
import argparse
from model import DissonanceClassifierBeforeAfter
from dataset import DissonanceDatasetwithbeforeafter
from dataset import DissonanceDatasetwithbeforeaftersep
from model import DissonanceClassifierBeforeAfterSep
from dataset import DissonanceWholeTweetWithSep
from model import DissonanceClassifierFullTweetSep
from model import DissonanceClassifierCrossAttn
from model import DissonanceClassifierCrossAttnWithTweet
from dataset import kialoDisagreeement
from model import DisagreementClassifierCrossAttn
from model import  kialoClassifierFullTweetSep
from model import BeforAfterWholeDissonance
from dataset import DissonanceDatasetBeforeafterWhole
from dataset import DissonanceDatasetTweetDus
from model import DissonanceClassifierTweetandDus
from collections import Counter
import optuna
logger = wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

def prepare_dataset(train_data_file, test_data_file, dev_data_file, args):
    #input_dataset = DissonanceDataset.from_files(train_data_file)
    if(args.architecture == 'basic'):
        train_dataset = DissonanceDataset.from_files(train_data_file)
        test_data = DissonanceDataset.from_files(test_data_file)
        dev_data = DissonanceDataset.from_files(dev_data_file)
    elif(args.architecture == 'before_after'):
        train_dataset = DissonanceDatasetwithbeforeafter.from_files(train_data_file)
        test_data = DissonanceDatasetwithbeforeafter.from_files(test_data_file)
        dev_data = DissonanceDatasetwithbeforeafter.from_files(dev_data_file)
    elif(args.architecture == 'before_after_sep'):
        train_dataset = DissonanceDatasetwithbeforeaftersep.from_files(train_data_file)
        test_data = DissonanceDatasetwithbeforeaftersep.from_files(test_data_file)
        dev_data = DissonanceDatasetwithbeforeaftersep.from_files(dev_data_file)
    elif(args.architecture == 'tweet_sep' or args.architecture=='cross_attn' or args.architecture=='cross_attn_tweet'):
        train_dataset = DissonanceWholeTweetWithSep.from_files(train_data_file)
        #print('input_dataset',input_dataset.dataset)
        test_data = DissonanceWholeTweetWithSep.from_files(test_data_file)
        dev_data = DissonanceWholeTweetWithSep.from_files(dev_data_file)
    elif(args.architecture == 'kialo_cross_attn'):
        train_dataset = kialoDisagreeement.from_files(train_data_file)
        #print('input_dataset',input_dataset.dataset)
        test_data = kialoDisagreeement.from_files(test_data_file)
        new_train_size = int(len(train_dataset) * 0.8)
        train_dataset, dev_data = torch.utils.data.random_split(train_dataset,

                                                             [new_train_size, len(train_dataset) - new_train_size])
    elif(args.architecture == 'kialo_sep'):
        train_dataset = kialoDisagreeement.from_files(train_data_file)
        #print('input_dataset',input_dataset.dataset)
        test_data = kialoDisagreeement.from_files(test_data_file)
        new_train_size = int(len(train_dataset) * 0.8)
        train_dataset, dev_data = torch.utils.data.random_split(train_dataset,
                                                             [new_train_size, len(train_dataset) - new_train_size])
    elif(args.architecture == 'before_after_whole'):
        train_dataset = DissonanceDatasetBeforeafterWhole.from_files(train_data_file)
        test_data = DissonanceDatasetBeforeafterWhole.from_files(test_data_file)
        dev_data = DissonanceDatasetBeforeafterWhole.from_files(dev_data_file)
    elif(args.architecture == 'tweet_batching'):
        train_dataset = DissonanceDatasetTweetDus.from_files(train_data_file)
        test_data = DissonanceDatasetTweetDus.from_files(test_data_file)
        dev_data = DissonanceDatasetTweetDus.from_files(dev_data_file)
    elif(args.architecture == 'pdtb_sep' or args.architecture == 'pdtb_cross_attn' ):
        full_dataset = kialoDisagreeement.from_files(train_data_file)
        #print('input_dataset',input_dataset.dataset)
        #test_data = kialoDisagreeement.from_files(test_data_file)
        new_train_size = int(len(full_dataset) * 0.6)
        new_test_size=(len(full_dataset) - new_train_size)//2
        print(new_test_size)
        train_dataset, dev_data, test_data = torch.utils.data.random_split(full_dataset,
                                                             [new_train_size, new_test_size, new_test_size])
    return train_dataset, test_data, dev_data

def prepare_output_dir(name_of_directory):
    os.system("mkdir -p output/")

    # save_name = str(max([int(i) for i in os.popen("ls /data1/vvaradarajan/output")], default=0) + 1)
    output_dir = os.path.join("output", name_of_directory)

    os.system(f"mkdir -p {output_dir}")

    return output_dir


def main(args):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1985)
    random.seed(1985)
    np.random.seed(1985)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["WANDB_PROJECT"] = args.name
    wandb.init(project=args.name, config={'upload_artifacts': False})

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"
    train_data, test_data, dev_data = prepare_dataset(args.train_file, args.test_file, args.dev_file, args)
    #Calculating Class Weights

    labels_full=[]
    list_train_data= list(train_data)
    for sample in list_train_data:
        labels_full.append(sample["labels"])
    # class_counts = dict(sorted(Counter(labels_full).items()))
    # print(class_counts)
    # total_samples = len(labels_full)
    # class_weights = torch.tensor([total_samples / (class_counts[label] * len(class_counts)) for label in class_counts])
    # print(class_weights)

    if (args.architecture == 'basic'):
        print('Model:',args.architecture)
        model = DissonanceClassifier().to(device)
    elif(args.architecture == 'before_after'):
        print('Model:', args.architecture)
        model = DissonanceClassifierBeforeAfter().to(device)
    elif(args.architecture == 'before_after_sep'):
        print('Model:', args.architecture)
        model = DissonanceClassifierBeforeAfterSep().to(device)
    elif(args.architecture == 'tweet_sep'):
        print('Model:', args.architecture)
        model = DissonanceClassifierFullTweetSep().to(device)
    elif(args.architecture == 'cross_attn'):
        print('Model:', args.architecture)
        model = DissonanceClassifierCrossAttn().to(device)
    elif(args.architecture == 'cross_attn_tweet'):
        print('Model:', args.architecture)
        model = DissonanceClassifierCrossAttnWithTweet(class_weights.to(device)).to(device)
    elif (args.architecture == 'kialo_cross_attn'):
        print('Model:', args.architecture)
        model = DisagreementClassifierCrossAttn().to(device)
    elif (args.architecture == 'kialo_sep'):
        print('Model:', args.architecture)
        model = kialoClassifierFullTweetSep().to(device)
    elif (args.architecture == 'before_after_whole'):
        print('Model:', args.architecture)
        model = BeforAfterWholeDissonance().to(device)
    elif (args.architecture == 'tweet_batching'):
        print('Model:', args.architecture)
        model = DissonanceClassifierTweetandDus().to(device)
    elif (args.architecture == 'pdtb_sep'):
        print('Model:', args.architecture)
        model = kialoClassifierFullTweetSep().to(device)
    elif (args.architecture == 'pdtb_cross_attn'):
        print('Model:', args.architecture)
        model = DisagreementClassifierCrossAttn().to(device)


    #print(args.train_file)

    #print(list(train_data))
    train(model, train_data, dev_data, test_data, args)

def calc_f1_avg(y_true, y_pred):
    f1_c1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0], average="macro")
    f1_c2 = f1_score(y_true=y_true, y_pred=y_pred, labels=[1], average="macro")
    f1_c3 = f1_score(y_true=y_true, y_pred=y_pred, labels=[2], average="macro")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    # also prec and recall for dissonance
    prec_dis = precision_score(y_true=y_true, y_pred=y_pred, labels=[1], average="macro")
    rec_dis = recall_score(y_true=y_true, y_pred=y_pred, labels=[1], average="macro")

    return np.mean([f1_c1, f1_c2]), np.mean([f1_c1, f1_c2, f1_c3]), f1_c1, f1_c2, f1_c3, acc, prec_dis, rec_dis

def calc_pdtb_f1_avg(y_true, y_pred):
    f1_c1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0], average="macro")
    f1_c2 = f1_score(y_true=y_true, y_pred=y_pred, labels=[1], average="macro")
    #f1_c3 = f1_score(y_true=y_true, y_pred=y_pred, labels=[2], average="macro")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    # also prec and recall for dissonance
    prec_dis = precision_score(y_true=y_true, y_pred=y_pred, labels=[1], average="macro")
    rec_dis = recall_score(y_true=y_true, y_pred=y_pred, labels=[1], average="macro")

    return np.mean([f1_c1, f1_c2]), np.mean([f1_c1, f1_c2]), f1_c1, f1_c2, acc, prec_dis, rec_dis

def train(model, train_data, dev_data, test_data, args):

    # print(list(train_data))

    def compute_metrics_agree(p):
        #print(p)
        part1, part2 = p
        #print(part1)
        #print(part2)
        pred=part1[0]
        labels=part1[1]
        metrics = dict()

        predictions = [np.exp(y_pred) / np.sum(np.exp(y_pred)) for y_pred in pred]
        disso_preds = [i[1] for i in predictions]

        _, f1_avg, f1_agree, f1_disagree, f1_na, acc, prec_dis, rec_dis = calc_f1_avg(y_true=labels,
                                                                                      y_pred=np.argmax(pred,
                                                                                                       axis=1))
        metrics.update({"f1_macro": f1_avg, "f1_consonance": f1_agree, "f1_dissonance": f1_disagree, "f1_na": f1_na,
                        "accuracy": acc, "prec_dis": prec_dis, "rec_dis": rec_dis})
        # y_true = [{0: 0, 1: 1, 2: 2}[i] for i in labels]
        # y_pred = [{0: 0, 1: 1, 2: 2}[i] for i in np.argmax(pred, axis=1)]
        # auc = roc_auc_score(labels, np.argmax(pred,axis=1))

        # auc_macro = roc_auc_score(labels, predictions, average="macro", multi_class="ovr")
        # auc_weighted = roc_auc_score(labels, predictions, average="weighted", multi_class="ovr")
        # metrics.update({"auc": auc, "auc_macro": auc_macro, "auc_weighted": auc_weighted})
        # metrics.update({"auc_prob": roc_auc_score(y_true, disso_preds)})
        y_true = [{0: 0, 1: 1, 2: 0}[i] for i in labels]
        y_pred = [{0: 0, 1: 1, 2: 0}[i] for i in np.argmax(pred, axis=1)]
        auc = roc_auc_score(y_true, y_pred)
        auc_macro = roc_auc_score(labels, predictions, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(labels, predictions, average="weighted", multi_class="ovr")
        metrics.update({"auc": auc, "auc_macro": auc_macro, "auc_weighted": auc_weighted})
        metrics.update({"auc_prob": roc_auc_score(y_true, disso_preds)})

        return metrics

    def compute_metrics_agree_pdtb(p):
        #print(p)
        part1, part2 = p
        #print(part1)
        #print(part2)
        pred=part1[0]
        labels=part1[1]
        metrics = dict()

        predictions = [np.exp(y_pred) / np.sum(np.exp(y_pred)) for y_pred in pred]
        disso_preds = [i[1] for i in predictions]

        _, f1_avg, f1_agree, f1_disagree, acc, prec_dis, rec_dis = calc_pdtb_f1_avg(y_true=labels,
                                                                                      y_pred=np.argmax(pred,
                                                                                                       axis=1))
        metrics.update({"f1_macro": f1_avg, "f1_consonance": f1_agree, "f1_dissonance": f1_disagree,
                        "accuracy": acc, "prec_dis": prec_dis, "rec_dis": rec_dis})
        # y_true = [{0: 0, 1: 1, 2: 2}[i] for i in labels]
        # y_pred = [{0: 0, 1: 1, 2: 2}[i] for i in np.argmax(pred, axis=1)]
        # auc = roc_auc_score(labels, np.argmax(pred,axis=1))

        # auc_macro = roc_auc_score(labels, predictions, average="macro", multi_class="ovr")
        # auc_weighted = roc_auc_score(labels, predictions, average="weighted", multi_class="ovr")
        # metrics.update({"auc": auc, "auc_macro": auc_macro, "auc_weighted": auc_weighted})
        # metrics.update({"auc_prob": roc_auc_score(y_true, disso_preds)})
        y_true = [{0: 0, 1: 1, 2: 0}[i] for i in labels]
        y_pred = [{0: 0, 1: 1, 2: 0}[i] for i in np.argmax(pred, axis=1)]
        auc = roc_auc_score(y_true, y_pred)
        auc_macro = roc_auc_score(labels, disso_preds, average="macro")
        auc_weighted = roc_auc_score(labels, disso_preds, average="weighted")
        metrics.update({"auc": auc, "auc_macro": auc_macro, "auc_weighted": auc_weighted})
        metrics.update({"auc_prob": roc_auc_score(y_true, disso_preds)})

        return metrics


    output_dir = prepare_output_dir(args.name)

    def objective(trial: optuna.Trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [1,4,8,16])
        num_epochs = trial.suggest_int('num_epochs', 1, 20)
        weight_decay = trial.suggest_loguniform("weight_decay", 4e-6, 0.01)
        optimizer = AdamW(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        training_args = TrainingArguments(
            output_dir=output_dir,
            report_to= 'wandb',
            logging_steps=100,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_total_limit=5,
            greater_is_better=True,
            seed=0,
            evaluation_strategy="steps",
            eval_steps=500,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            metric_for_best_model="eval_f1_dissonance",
            save_steps=1000,
            remove_unused_columns=False,
            weight_decay=weight_decay,
            load_best_model_at_end=True
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset= train_data,
            eval_dataset=dev_data,
            data_collator=model.data_collator,
            compute_metrics=compute_metrics_agree,
            optimizers=(optimizer,None),
            #callbacks=[early_stopping],
        )
        trainer.train()
        eval_results = trainer.evaluate()
        evaluate(trainer, test_data)
        wandb.finish()

        return eval_results['eval_f1_dissonance']

    optimizer = AdamW(model.parameters(), weight_decay=0.0000058, lr=0.00001179)
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to='wandb',
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        greater_is_better=True,
        seed=0,
        evaluation_strategy="steps",
        eval_steps=1000,
        num_train_epochs=10,
        learning_rate=5e-5,#0.0000058,#0.000004037,
        metric_for_best_model="eval_f1_dissonance",
        save_steps=1000,
        remove_unused_columns=False,
        weight_decay=5e-4#0.00005939
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=model.data_collator,
        compute_metrics=compute_metrics_agree_pdtb,
        optimizers=(optimizer, None),
        #callbacks=[early_stopping],
    )
    trainer.train()
    trainer.evaluate()
    results_test = evaluate(trainer, test_data)
    print(results_test)
    # study = optuna.create_study(study_name='hyper-parameter-search', direction='maximize')
    # study.optimize(func=objective, n_trials=25)
    # print(study.best_params)
    # print(study.best_value)
    # print(study.best_trial)


    # [optional] finish the wandb run, necessary in notebooks


def evaluate(trainer, test_data):
    log = dict()
    #trainer.eval_dataset=test_data
    test_output = trainer.predict(test_data)
    #print(test_data)
    #print('test output-->')
    #part1,part2=test_output
    #print(test_output.metrics)
    log.update(test_output.metrics)
    print(test_output.metrics)

    # y_pred = [y_pred.argmax(0) for inst, y_pred in zip(test_data, test_output.predictions)]
    # predictions = [np.exp(y_pred) / np.sum(np.exp(y_pred)) for inst, y_pred in
    #                zip(test_data, test_output.predictions)]
    #
    # y_true = [inst["label"] for inst, y_pred in zip(test_data, test_output.predictions)]
    # _, eval_f1_avg, eval_f1_agree, eval_f1_disagree, eval_f1_na, eval_acc, eval_prec_dis, eval_recall_dis = calc_f1_avg(
    #     y_pred, y_true)
    # auc_macro = roc_auc_score(y_true, predictions, average="macro", multi_class="ovr")
    # auc_weighted = roc_auc_score(y_true, predictions, average="weighted", multi_class="ovr")
    #
    # y_true = [{0: 0, 1: 1, 2: 0}[i] for i in y_true]
    # y_pred = [{0: 0, 1: 1, 2: 0}[i] for i in y_pred]
    # eval_auc = roc_auc_score(y_true, y_pred)
    # eval_auc_prob = roc_auc_score(y_true, [disso_preds[1] for disso_preds in test_output.predictions])

    # log.update({
    #     f"test/f1_macro": eval_f1_avg,
    #     f"test/f1_consonance": eval_f1_agree,
    #     f"test/f1_dissonance": eval_f1_disagree,
    #     f"test/f1_na": eval_f1_na,
    #     f"test/acc": eval_acc,
    #     f"test/auc": eval_auc,
    #     f"test/auc_macro": auc_macro,
    #     f"test/auc_weighted": auc_weighted,
    #     f"test/auc_prob": eval_auc_prob,
    #     f"test/disso_precision": eval_prec_dis,
    #     f"test/disso_recall": eval_recall_dis,
    # })
    # print(log)
    logger.log(log)

    return test_output.metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-trainfile', '--train-file', type=str, required=True,
        help="Input datasets")
    parser.add_argument(
        '-testfile', '--test-file', type=str, required=True,
        help="Test datasets")
    parser.add_argument(
        '-devfile', '--dev-file', type=str, required=True,
        help="Dev datasets")
    parser.add_argument(
        '-nm', '--name', type=str, required=True, help='Name of the experiment.')


    parser.add_argument(
        '-arc', '--architecture', type=str, default="sep",
        choices=["basic", "before_after", "before_after_sep","tweet_sep","cross_attn", "cross_attn_tweet","kialo_cross_attn", "kialo_sep", 'before_after_whole','tweet_batching', 'pdtb_sep','pdtb_cross_attn'],
        help="Architecture type.")
    args = parser.parse_args()
    # test_d=DissonanceWholeTweetWithSep.from_files('test.json')
    # print(test_d[10])
    # model= DissonanceClassifierFullTweetSep()
    # model.data_collator(inst=test_d)
    main(args)
    # dataset_init = DissonanceDataset.from_files('test.json')
    # model = DissonanceClassifier()
    # formatted = model.data_collator(list(dataset_init))
    # print(formatted)
    # model.forward(formatted['context'],
    #               formatted['list_of_pairs_enc'],
    #               formatted['labels'])
