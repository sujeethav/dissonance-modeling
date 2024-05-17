import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class DissonanceClassifier(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels,
                                                                            hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 2, 500), torch.nn.ReLU(True),
                                                 torch.nn.Linear(500, self.num_labels))

    def forward(self, context, list_of_pairs_enc, labels):
        #print('coming from forward',len(context['input_ids']),len(list_of_pairs_enc),len(labels))
        tweets_embeddings = self.enc_model(input_ids=context["input_ids"], output_hidden_states=True).hidden_states[-1]
        #tweets_embeddings=torch.transpose(tweets_embeddings, 0, 1)[0]
        tweets_embeddings=torch.mean(tweets_embeddings, dim=1)
        #print(tweets_embeddings.shape)
        n=len(list_of_pairs_enc)
        loss=torch.tensor(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits=torch.tensor([]).to(device)
        for i in range(n):
            pair_embeddings = self.enc_model(input_ids=list_of_pairs_enc[i]["input_ids"], output_hidden_states=True).hidden_states[-1]
            #print(pair_embeddings.shape)
            pair_embeddings=torch.mean(pair_embeddings, dim=1)
            #print(tweets_embeddings[i].unsqueeze(0).shape)
            #print(pair_embeddings.shape)
            part1=tweets_embeddings[i].unsqueeze(0).expand(pair_embeddings.shape[0], -1)
            #print(part1)
            combined=torch.cat([part1,pair_embeddings],dim=1)
            #print(combined[0])
            output_logits=self.linear_classifier(combined).to(device)

            loss = loss + self.loss_fn(output_logits, labels[i].to(device))
            # print(loss)
            logits = torch.cat((logits,output_logits),dim=0)
            #print(output_logits.shape, torch.tensor(labels[i]).shape)



        # print("After the iteration",loss)
        return {
                "loss": loss,
                "logits": logits,
                "labels":torch.cat(labels, dim=0)
            }


    def data_collator(self, inst):
        # print(inst)

        # full_tweets = [i["tweet"].replace("<", "$123$").replace(" > ", "$321$") for i in inst] #" ".join([i.strip() for i in (" ".join([each.strip() for each in inst['tweet'].split("<")])).split(">")])
        full_tweets =  []
        diss_pairs_combined_enc = []
        labels=[]
        for i in inst:
            full_tweets.append(i['tweet'])

            pairs=np.asarray(i['list_pairs'])
            #print(pairs)
            concatenated_column = np.core.defchararray.add( pairs[:,0].astype(str).tolist(), " </s> ")
            concatenated_column = np.core.defchararray.add(concatenated_column, pairs[:,1].astype(str).tolist())
            concatenated_column_encoding=self.enc_tok(concatenated_column.tolist(),
                              return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
            diss_pairs_combined_enc.append(concatenated_column_encoding)
            #stacked = np.column_stack((concatenated_column_encoding, labels))
            #list_of_pairs.append(stacked)
            labels.append(torch.tensor(i['labels']))


        full_tweets_enc = self.enc_tok(full_tweets,
                               return_tensors="pt", padding=True,
                               truncation=True, max_length=512)


        #print('coming from collator', len(full_tweets_enc.input_ids), len(diss_pairs_combined_enc), len(labels))
        return {"context": full_tweets_enc,
                "list_of_pairs_enc": diss_pairs_combined_enc,
                "labels":labels}

    def get_embeddings(self, inst):
        inputs = self.data_collator(inst)
        context = inputs["context"]
        with torch.no_grad():
            outputs = self.enc_model.roberta(
                input_ids=context["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu"),
                output_hidden_states=True)
            outputs = outputs.hidden_states[-2]
            outputs = torch.transpose(outputs, 0, 1)[0]
        return outputs


class DissonanceClassifierBeforeAfter(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels,
                                                                            hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 4, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))

    def forward(self, context, list_of_pairs_enc, labels, before_du1_enc, after_du2_enc):
        # print('coming from forward',len(context['input_ids']),len(list_of_pairs_enc),len(labels))
        tweets_embeddings = self.enc_model(input_ids=context["input_ids"], output_hidden_states=True).hidden_states[-1]
        # tweets_embeddings=torch.transpose(tweets_embeddings, 0, 1)[0]
        tweets_embeddings = torch.mean(tweets_embeddings, dim=1)
        # print(tweets_embeddings.shape)
        n = len(list_of_pairs_enc)
        loss = torch.tensor(0)
        device = torch.device("cpu")
        logits = torch.tensor([])  # .to(device)
        for i in range(n):
            pair_embeddings = \
            self.enc_model(input_ids=list_of_pairs_enc[i]["input_ids"], output_hidden_states=True).hidden_states[-1]
            # print(pair_embeddings.shape)
            pair_embeddings = torch.mean(pair_embeddings, dim=1)
            before_du1_embedding = self.enc_model(input_ids=before_du1_enc[i]["input_ids"], output_hidden_states=True).hidden_states[-1]
            before_du1_embedding = torch.mean(before_du1_embedding, dim=1)
            # print(before_du1_embedding.size())

            after_du2_embedding = \
            self.enc_model(input_ids=after_du2_enc[i]["input_ids"], output_hidden_states=True).hidden_states[-1]
            after_du2_embedding = torch.mean(after_du2_embedding, dim=1)
            # print(tweets_embeddings[i].unsqueeze(0).shape)
            # print(pair_embeddings.shape)
            part1 = tweets_embeddings[i].unsqueeze(0).expand(pair_embeddings.shape[0], -1)
            # print(part1)
            combined = torch.cat([part1, pair_embeddings, before_du1_embedding, after_du2_embedding], dim=1)
            # print(combined[0])
            output_logits = self.linear_classifier(combined).to(device)

            loss = loss + self.loss_fn(output_logits, labels[i].to(device))
            # print(loss)
            logits = torch.cat((logits, output_logits), dim=0)
            # print(output_logits.shape, torch.tensor(labels[i]).shape)

        # print("After the iteration",loss)
        return {
            "loss": loss,
            "logits": logits,
            "labels": torch.cat(labels, dim=0)
        }

    def data_collator(self, inst):

        # full_tweets = [i["tweet"].replace("<", "$123$").replace(" > ", "$321$") for i in inst] #" ".join([i.strip() for i in (" ".join([each.strip() for each in inst['tweet'].split("<")])).split(">")])
        full_tweets = []
        diss_pairs_combined_enc = []
        labels = []
        before_du1_enc=[]
        after_du2_enc=[]
        for i in inst:
            full_tweets.append(i['tweet'])

            pairs = np.asarray(i['list_pairs'])
            # print(pairs)
            concatenated_column = np.core.defchararray.add(pairs[:, 0].astype(str).tolist(), " </s> ")
            concatenated_column = np.core.defchararray.add(concatenated_column, pairs[:, 1].astype(str).tolist())
            concatenated_column_encoding = self.enc_tok(concatenated_column.tolist(),
                                                        return_tensors="pt", padding=True,
                                                        truncation=True, max_length=512)
            before_du1_sentences_enc = self.enc_tok(i['before_du1'],
                                                        return_tensors="pt", padding=True,
                                                        truncation=True, max_length=512)
            after_du2_sentences_enc = self.enc_tok(i['after_du2'],
                                               return_tensors="pt", padding=True,
                                               truncation=True, max_length=512)
            before_du1_enc.append(before_du1_sentences_enc)
            after_du2_enc.append(after_du2_sentences_enc)

            diss_pairs_combined_enc.append(concatenated_column_encoding)
            # stacked = np.column_stack((concatenated_column_encoding, labels))
            # list_of_pairs.append(stacked)
            labels.append(torch.tensor(i['labels']))

        full_tweets_enc = self.enc_tok(full_tweets,
                                       return_tensors="pt", padding=True,
                                       truncation=True, max_length=512)

        # print('coming from collator', len(full_tweets_enc.input_ids), len(diss_pairs_combined_enc), len(labels))
        return {"context": full_tweets_enc,
                "list_of_pairs_enc": diss_pairs_combined_enc,
                "labels": labels,
                "before_du1_enc": before_du1_enc,
                "after_du2_enc": after_du2_enc}



class DissonanceClassifierBeforeAfterSep(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels,
                                                                            hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 3, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))

    def forward(self, context, list_of_pairs_enc, labels, before_after_sep_enc):
        # print('coming from forward',len(context['input_ids']),len(list_of_pairs_enc),len(labels))
        # print(context["input_ids"])
        tweets_embeddings = self.enc_model(input_ids=context["input_ids"], output_hidden_states=True).hidden_states[-1]
        # tweets_embeddings=torch.transpose(tweets_embeddings, 0, 1)[0]
        tweets_embeddings = torch.mean(tweets_embeddings, dim=1)
        # print(tweets_embeddings.shape)
        n = len(list_of_pairs_enc)
        loss = torch.tensor(0)
        device = torch.device("cpu")
        logits = torch.tensor([])  # .to(device)
        for i in range(n):
            pair_embeddings = \
            self.enc_model(input_ids=list_of_pairs_enc[i]["input_ids"], output_hidden_states=True).hidden_states[-1]
            # print(pair_embeddings.shape)
            pair_embeddings = torch.mean(pair_embeddings, dim=1)
            before_after_sep_embedding = self.enc_model(input_ids=before_after_sep_enc[i]["input_ids"], output_hidden_states=True).hidden_states[-1]
            before_after_sep_embedding = torch.mean(before_after_sep_embedding, dim=1)

            part1 = tweets_embeddings[i].unsqueeze(0).expand(pair_embeddings.shape[0], -1)

            combined = torch.cat([part1, pair_embeddings, before_after_sep_embedding], dim=1)

            output_logits = self.linear_classifier(combined).to(device)

            loss = loss + self.loss_fn(output_logits, labels[i].to(device))
            # print(loss)
            logits = torch.cat((logits, output_logits), dim=0)


        # print("After the iteration",loss)
        return {
            "loss": loss,
            "logits": logits,
            "labels": torch.cat(labels, dim=0)
        }

    def data_collator(self, inst):


        full_tweets = []
        diss_pairs_combined_enc = []
        labels = []
        before_after_sep_enc=[]
        for i in inst:
            full_tweets.append(i['tweet'])

            pairs = np.asarray(i['list_pairs'])
            # print(pairs)
            concatenated_column = np.core.defchararray.add(pairs[:, 0].astype(str).tolist(), " </s> ")
            concatenated_column = np.core.defchararray.add(concatenated_column, pairs[:, 1].astype(str).tolist())
            concatenated_column_encoding = self.enc_tok(concatenated_column.tolist(),
                                                        return_tensors="pt", padding=True,
                                                        truncation=True, max_length=512)
            before_after_sep_sent_enc = self.enc_tok(i['before_after_sep'],
                                                        return_tensors="pt", padding=True,
                                                        truncation=True, max_length=512)

            before_after_sep_enc.append(before_after_sep_sent_enc)

            diss_pairs_combined_enc.append(concatenated_column_encoding)
            # stacked = np.column_stack((concatenated_column_encoding, labels))
            # list_of_pairs.append(stacked)
            labels.append(torch.tensor(i['labels']))

        full_tweets_enc = self.enc_tok(full_tweets,
                                       return_tensors="pt", padding=True,
                                       truncation=True, max_length=512)

        # print('coming from collator', len(full_tweets_enc.input_ids), len(diss_pairs_combined_enc), len(labels))
        return {"context": full_tweets_enc,
                "list_of_pairs_enc": diss_pairs_combined_enc,
                "labels": labels,
                "before_after_sep_enc": before_after_sep_enc}


class DissonanceClassifierFullTweetSep(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name, sep_token='$123$')
        self.enc_tok.add_tokens(["$123$"])
        self.enc_model.resize_token_embeddings(len(self.enc_tok))

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 2, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))

    def forward(self, tweet, diss_pairs_ind, labels):

        tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
                                           attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(tweets_embeddings.shape)
        n = len(diss_pairs_ind)

        enc_logits = []  # .to(device)
        for i in range(n):
            one_tweet_enconding=tweets_embeddings[i]
            #print(one_tweet_enconding.size())

            du1_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]], dim=0)
            du2_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]], dim=0)

            one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
            combined = torch.cat([du1_embedding, du2_embedding], dim=0)
            enc_logits.append(combined)
            #print('here')
        enc_logits=torch.stack(enc_logits)
        #print('enc_shape', enc_logits.size())
        output_logits = self.linear_classifier(enc_logits).to(device)

        loss =self.loss_fn(output_logits.to(device), labels.to(device))

        return {
            "loss": loss.to(device),
            "logits": output_logits.to(device),
            "labels":labels.to(device)
        }

    def data_collator(self, inst):

        #print(inst)
        diss_pairs_ind = []
        labels =torch.tensor([])
        tweets = []
        for i in inst:
            tweets.append(i['tweet'])
            labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)

        full_tweets_enc = self.enc_tok(tweets,
                                        return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)
        #print(full_tweets_enc['input_ids'][0], inst[0])
        #print(full_tweets_enc['input_ids'][1],inst[1])

        sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]

        diss_pairs_ind = [ ]
        for i in range(len(inst)):
            sep_indices_twt = sep_indices[i]
            #print(i,sep_indices_twt)
            du_pairs = inst[i]['du_pairs_index']
            len_sep_indices_twt = len(sep_indices_twt)
            #print(du_pairs)
            pair_dus=[]
            for p in du_pairs:
                temp = []
                if p ==0:
                    temp.append(0)
                    temp.append(sep_indices_twt[0])
                elif p==len_sep_indices_twt:
                    temp.append(sep_indices_twt[-1]+1)
                    temp.append(len(full_tweets_enc['input_ids'][i]))
                else:
                    temp.append(sep_indices_twt[p-1]+1)
                    temp.append(sep_indices_twt[p])
                pair_dus.append(temp)
            diss_pairs_ind.append(pair_dus)
        return {"tweet": full_tweets_enc,
                "diss_pairs_ind": torch.tensor(diss_pairs_ind),
                "labels": labels.long()
                }

class DissonanceClassifierCrossAttn(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.CrossEntropyLoss()#(weight=class_weights)
        #print(class_weights)
        model_name = "roberta-base"
        self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name, sep_token='$123$')
        self.enc_tok.add_tokens(["$123$"])
        self.enc_model.resize_token_embeddings(len(self.enc_tok))
        self.input_size = 768
        self.linear_q = torch.nn.Linear( self.input_size,  self.input_size)
        self.linear_k = torch.nn.Linear( self.input_size,  self.input_size)
        self.linear_v = torch.nn.Linear( self.input_size,  self.input_size)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))


    def forward(self, tweet, diss_pairs_ind, labels):

        tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
                                           attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(tweets_embeddings.shape)
        n = len(diss_pairs_ind)

        enc_logits = []  # .to(device)
        du1_batch_embeddings = torch.tensor([]).to(device)
        du2_batch_embeddings = torch.tensor([])
        for i in range(n):
            one_tweet_enconding=tweets_embeddings[i]
            #print(one_tweet_enconding.size())

            du1_embedding = one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]]
            du2_embedding = one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]]
            #print("du1_embedding",du1_embedding.size())
            #one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
            #combined = torch.cat([du1_embedding, du2_embedding], dim=0)
            #du1_batch_embeddings= torch.cat((du1_batch_embeddings, du1_embedding.to(device)), dim=0)#append(du1_embedding)
            q = self.linear_q(du1_embedding)  # query vector
            k = self.linear_k(du2_embedding)  # key vector
            v = self.linear_v(du2_embedding)
            #print('q',q.size())
            #print('k', k.size())
            attn_scores = q @ k.T / (self.input_size ** 0.5) #torch.bmm(q, k.transpose(0, 1)) / (self.input_size ** 0.5)
            attn_scores = torch.softmax(attn_scores, dim=-1)
            attn_weights_output = attn_scores @ v
            attn_weights_output_mean= torch.mean(attn_weights_output,dim=0)
            enc_logits.append(attn_weights_output_mean)

        #print('enc_shape', enc_logits.size())
        enc_logits=torch.stack(enc_logits)
        output_logits = self.linear_classifier(enc_logits).to(device)

        loss =self.loss_fn(output_logits.to(device), labels.to(device))

        return {
            "loss": loss.to(device),
            "logits": output_logits.to(device),
            "labels":labels.to(device)
        }

    def data_collator(self, inst):

        #print(inst)
        diss_pairs_ind = []
        labels =torch.tensor([])
        tweets = []
        for i in inst:
            tweets.append(i['tweet'])
            labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)

        full_tweets_enc = self.enc_tok(tweets,
                                        return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)
        #print(full_tweets_enc['input_ids'][0], inst[0])
        #print(full_tweets_enc['input_ids'][1],inst[1])

        sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]

        diss_pairs_ind = [ ]
        for i in range(len(inst)):
            sep_indices_twt = sep_indices[i]
            #print(i,sep_indices_twt)
            du_pairs = inst[i]['du_pairs_index']
            len_sep_indices_twt = len(sep_indices_twt)
            #print(du_pairs)
            pair_dus=[]
            for p in du_pairs:
                temp = []
                if p ==0:
                    temp.append(0)
                    temp.append(sep_indices_twt[0])
                elif p==len_sep_indices_twt:
                    temp.append(sep_indices_twt[-1]+1)
                    temp.append(len(full_tweets_enc['input_ids'][i]))
                else:
                    temp.append(sep_indices_twt[p-1]+1)
                    temp.append(sep_indices_twt[p])
                pair_dus.append(temp)
            diss_pairs_ind.append(pair_dus)
        return {"tweet": full_tweets_enc,
                "diss_pairs_ind": torch.tensor(diss_pairs_ind),
                "labels": labels.long()
                }


class DissonanceClassifierCrossAttnWithTweet(torch.nn.Module):

    def __init__(self, class_weights):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        #print(class_weights)
        model_name = "roberta-base"
        self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)
        #self.enc_tok.add_tokens(["$123$"])
        self.enc_model.resize_token_embeddings(len(self.enc_tok))
        self.input_size = 768
        self.linear_q = torch.nn.Linear( self.input_size,  self.input_size)
        self.linear_k = torch.nn.Linear( self.input_size,  self.input_size)
        self.linear_v = torch.nn.Linear( self.input_size,  self.input_size)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768*2, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))


    def forward(self, tweet, diss_pairs_ind, labels):

        tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
                                           attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(tweets_embeddings.shape)
        n = len(diss_pairs_ind)

        enc_logits = []  # .to(device)
        du1_batch_embeddings = torch.tensor([]).to(device)
        du2_batch_embeddings = torch.tensor([])
        for i in range(n):
            one_tweet_enconding=tweets_embeddings[i]
            #print(one_tweet_enconding.size())

            du1_embedding = one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]]
            du2_embedding = one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]]
            #print("du1_embedding",du1_embedding.size())
            one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
            #combined = torch.cat([du1_embedding, du2_embedding], dim=0)
            #du1_batch_embeddings= torch.cat((du1_batch_embeddings, du1_embedding.to(device)), dim=0)#append(du1_embedding)
            q = self.linear_q(du1_embedding)  # query vector
            k = self.linear_k(du2_embedding)  # key vector
            v = self.linear_v(du2_embedding)
            #print('q',q.size())
            #print('k', k.size())
            attn_scores = q @ k.T / (self.input_size ** 0.5) #torch.bmm(q, k.transpose(0, 1)) / (self.input_size ** 0.5)
            attn_scores = torch.softmax(attn_scores, dim=-1)
            attn_weights_output = attn_scores @ v
            attn_weights_output_mean= torch.mean(attn_weights_output,dim=0)
            combined = torch.cat([one_tweet_embedding, attn_weights_output_mean], dim=0)
            enc_logits.append(combined)

        #print('enc_shape', enc_logits.size())
        enc_logits=torch.stack(enc_logits)
        output_logits = self.linear_classifier(enc_logits).to(device)

        loss =self.loss_fn(output_logits.to(device), labels.to(device))

        return {
            "loss": loss.to(device),
            "logits": output_logits.to(device),
            "labels":labels.to(device)
        }

    def data_collator(self, inst):

        #print(inst)
        diss_pairs_ind = []
        labels =torch.tensor([])
        tweets = []
        for i in inst:
            tweets.append(i['tweet'])
            labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)

        full_tweets_enc = self.enc_tok(tweets,
                                        return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)
        #print(full_tweets_enc['input_ids'][0], inst[0])
        #print(full_tweets_enc['input_ids'][1],inst[1])

        sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]

        diss_pairs_ind = [ ]
        for i in range(len(inst)):
            sep_indices_twt = sep_indices[i]
            #print(i,sep_indices_twt)
            du_pairs = inst[i]['du_pairs_index']
            len_sep_indices_twt = len(sep_indices_twt)
            #print(du_pairs)
            pair_dus=[]
            for p in du_pairs:
                temp = []
                if p ==0:
                    temp.append(0)
                    temp.append(sep_indices_twt[0])
                elif p==len_sep_indices_twt:
                    temp.append(sep_indices_twt[-1]+1)
                    temp.append(len(full_tweets_enc['input_ids'][i]))
                else:
                    temp.append(sep_indices_twt[p-1]+1)
                    temp.append(sep_indices_twt[p])
                pair_dus.append(temp)
            diss_pairs_ind.append(pair_dus)
        return {"tweet": full_tweets_enc,
                "diss_pairs_ind": torch.tensor(diss_pairs_ind),
                "labels": labels.long()
                }

class DisagreementClassifierCrossAttn(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.CrossEntropyLoss()
        #print(class_weights)
        model_name = "roberta-base"
        self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)
        #self.enc_tok.add_tokens(["$123$"])
        self.enc_model.resize_token_embeddings(len(self.enc_tok))
        self.input_size = 768
        self.linear_q = torch.nn.Linear( self.input_size,  self.input_size)
        self.linear_k = torch.nn.Linear( self.input_size,  self.input_size)
        self.linear_v = torch.nn.Linear( self.input_size,  self.input_size)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))


    def forward(self, tweet, diss_pairs_ind, labels):

        tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
                                           attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(tweets_embeddings.shape)
        n = len(diss_pairs_ind)

        enc_logits = []  # .to(device)
        du1_batch_embeddings = torch.tensor([]).to(device)
        du2_batch_embeddings = torch.tensor([])
        for i in range(n):
            one_tweet_enconding=tweets_embeddings[i]
            #print(one_tweet_enconding.size())

            du1_embedding = one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]]
            du2_embedding = one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]]
            #print("du1_embedding",du1_embedding.size())
            #one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
            #combined = torch.cat([du1_embedding, du2_embedding], dim=0)
            #du1_batch_embeddings= torch.cat((du1_batch_embeddings, du1_embedding.to(device)), dim=0)#append(du1_embedding)
            q = self.linear_q(du1_embedding)  # query vector
            k = self.linear_k(du2_embedding)  # key vector
            v = self.linear_v(du2_embedding)
            #print('q',q.size())
            #print('k', k.size())
            attn_scores = q @ k.T / (self.input_size ** 0.5) #torch.bmm(q, k.transpose(0, 1)) / (self.input_size ** 0.5)
            attn_scores = torch.softmax(attn_scores, dim=-1)
            attn_weights_output = attn_scores @ v
            attn_weights_output_mean= torch.mean(attn_weights_output,dim=0)
            #combined = torch.cat([one_tweet_embedding, attn_weights_output_mean], dim=0)
            enc_logits.append(attn_weights_output_mean)

        #print('enc_shape', enc_logits.size())
        enc_logits=torch.stack(enc_logits)
        output_logits = self.linear_classifier(enc_logits).to(device)

        loss =self.loss_fn(output_logits.to(device), labels.to(device))

        return {
            "loss": loss.to(device),
            "logits": output_logits.to(device),
            "labels":labels.to(device)
        }

    def data_collator(self, inst):

        #print(inst)
        diss_pairs_ind = []
        labels =torch.tensor([])
        tweets = []
        for i in inst:
            tweets.append(i['tweet'])
            labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)

        full_tweets_enc = self.enc_tok(tweets,
                                        return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)
        #print(full_tweets_enc['input_ids'][0], inst[0])
        #print(full_tweets_enc['input_ids'][1],inst[1])

        sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]
        #print(full_tweets_enc)
        #print(sep_indices)
        diss_pairs_ind = [ ]
        for i in range(len(inst)):
            sep_indices_twt = sep_indices[i]
            #print(i,sep_indices_twt)
            du_pairs = [0,1]
            len_sep_indices_twt = len(sep_indices_twt)
            #print(du_pairs)
            pair_dus=[]
            for p in du_pairs:
                temp = []
                if p ==0:
                    temp.append(0)
                    temp.append(sep_indices_twt[0])
                elif p==len_sep_indices_twt:
                    temp.append(sep_indices_twt[-1]+1)
                    temp.append(len(full_tweets_enc['input_ids'][i]))
                else:
                    temp.append(sep_indices_twt[p-1]+1)
                    temp.append(sep_indices_twt[p])
                pair_dus.append(temp)
            diss_pairs_ind.append(pair_dus)
        #print(diss_pairs_ind)
        return {"tweet": full_tweets_enc,
                "diss_pairs_ind": torch.tensor(diss_pairs_ind),
                "labels": labels.long()
                }


class kialoClassifierFullTweetSep(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 2
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name, sep_token='$123$')
        self.enc_tok.add_tokens(["$123$"])
        self.enc_model.resize_token_embeddings(len(self.enc_tok))

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 2, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))

    def forward(self, tweet, diss_pairs_ind, labels):

        tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
                                           attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(tweets_embeddings.shape)
        n = len(diss_pairs_ind)

        enc_logits = []  # .to(device)
        for i in range(n):
            one_tweet_enconding=tweets_embeddings[i]
            #print(one_tweet_enconding.size())

            du1_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]], dim=0)
            du2_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]], dim=0)

            #one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
            combined = torch.cat([du1_embedding, du2_embedding], dim=0)
            enc_logits.append(combined)
            #print('here')
        enc_logits=torch.stack(enc_logits)
        # print('enc_shape', enc_logits.size())
        output_logits = self.linear_classifier(enc_logits).to(device)

        loss =self.loss_fn(output_logits.to(device), labels.to(device))

        return {
            "loss": loss.to(device),
            "logits": output_logits.to(device),
            "labels":labels.to(device)
        }

    def data_collator(self, inst):

        #print(inst)
        diss_pairs_ind = []
        labels =torch.tensor([])
        tweets = []
        for i in inst:
            tweets.append(i['tweet'])
            labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)

        full_tweets_enc = self.enc_tok(tweets,
                                        return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)
        # print(full_tweets_enc['input_ids'][0], inst[0])
        # print(full_tweets_enc['input_ids'][1],inst[1])

        sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]

        diss_pairs_ind = [ ]
        for i in range(len(inst)):
            sep_indices_twt = sep_indices[i]
            #print(i,sep_indices_twt)
            du_pairs = [0,1]#inst[i]['du_pairs_index']
            len_sep_indices_twt = len(sep_indices_twt)
            #print(du_pairs)
            pair_dus=[]
            for p in du_pairs:
                temp = []
                if p ==0:
                    temp.append(0)
                    temp.append(sep_indices_twt[0])
                elif p==len_sep_indices_twt:
                    temp.append(sep_indices_twt[-1]+1)
                    temp.append(len(full_tweets_enc['input_ids'][i]))
                else:
                    temp.append(sep_indices_twt[p-1]+1)
                    temp.append(sep_indices_twt[p])
                pair_dus.append(temp)
            diss_pairs_ind.append(pair_dus)
        return {"tweet": full_tweets_enc,
                "diss_pairs_ind": torch.tensor(diss_pairs_ind),
                "labels": labels.long()
                }


# class kialoClassifierFullTweetSep(torch.nn.Module):
#
#     def __init__(self, *args):
#         super().__init__()
#         # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
#         self.num_labels = 3
#         self.with_context = True
#         self.dropout = 0.1
#
#         self.loss_fn = torch.nn.CrossEntropyLoss()
#         model_name = "roberta-base"
#         self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
#         self.enc_tok = AutoTokenizer.from_pretrained(model_name, sep_token='$123$')
#         self.enc_tok.add_tokens(["$123$"])
#         self.enc_model.resize_token_embeddings(len(self.enc_tok))
#
#         self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 2, 500), torch.nn.ReLU(True),
#                                                      torch.nn.Linear(500, self.num_labels))
#
#     def forward(self, tweet, diss_pairs_ind, labels):
#
#         tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
#                                            attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # print(tweets_embeddings.shape)
#         n = len(diss_pairs_ind)
#
#         enc_logits = []  # .to(device)
#         for i in range(n):
#             one_tweet_enconding=tweets_embeddings[i]
#             #print(one_tweet_enconding.size())
#
#             du1_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]], dim=0)
#             du2_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]], dim=0)
#
#             #one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
#             combined = torch.cat([du1_embedding, du2_embedding], dim=0)
#             enc_logits.append(combined)
#             #print('here')
#         enc_logits=torch.stack(enc_logits)
#         # print('enc_shape', enc_logits.size())
#         output_logits = self.linear_classifier(enc_logits).to(device)
#
#         loss =self.loss_fn(output_logits.to(device), labels.to(device))
#
#         return {
#             "loss": loss.to(device),
#             "logits": output_logits.to(device),
#             "labels":labels.to(device)
#         }
#
#     def data_collator(self, inst):
#
#         #print(inst)
#         diss_pairs_ind = []
#         labels =torch.tensor([])
#         tweets = []
#         for i in inst:
#             tweets.append(i['tweet'])
#             labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)
#
#         full_tweets_enc = self.enc_tok(tweets,
#                                         return_tensors="pt", padding=True,
#                                         truncation=True, max_length=512)
#         # print(full_tweets_enc['input_ids'][0], inst[0])
#         # print(full_tweets_enc['input_ids'][1],inst[1])
#
#         sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]
#
#         diss_pairs_ind = [ ]
#         for i in range(len(inst)):
#             sep_indices_twt = sep_indices[i]
#             #print(i,sep_indices_twt)
#             du_pairs = [0,1]#inst[i]['du_pairs_index']
#             len_sep_indices_twt = len(sep_indices_twt)
#             #print(du_pairs)
#             pair_dus=[]
#             for p in du_pairs:
#                 temp = []
#                 if p ==0:
#                     temp.append(0)
#                     temp.append(sep_indices_twt[0])
#                 elif p==len_sep_indices_twt:
#                     temp.append(sep_indices_twt[-1]+1)
#                     temp.append(len(full_tweets_enc['input_ids'][i]))
#                 else:
#                     temp.append(sep_indices_twt[p-1]+1)
#                     temp.append(sep_indices_twt[p])
#                 pair_dus.append(temp)
#             diss_pairs_ind.append(pair_dus)
#         return {"tweet": full_tweets_enc,
#                 "diss_pairs_ind": torch.tensor(diss_pairs_ind),
#                 "labels": labels.long()
#                 }


class BeforAfterWholeDissonance(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name, sep_token='$123$')
        self.enc_tok.add_tokens(["$123$"])
        self.enc_model.resize_token_embeddings(len(self.enc_tok))

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 4, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))

    def forward(self, tweet, diss_pairs_ind, labels):

        tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
                                           attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(tweets_embeddings.shape)
        n = len(diss_pairs_ind)

        enc_logits = []  # .to(device)
        for i in range(n):
            one_tweet_enconding=tweets_embeddings[i]
            #print(one_tweet_enconding.size())
            only_tweet=torch.mean(one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]], dim=0)
            du1_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]], dim=0)
            du2_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][2][0]:diss_pairs_ind[i][2][1]], dim=0)
            before_after_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][3][0]:diss_pairs_ind[i][3][1]], dim=0)

            #one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
            combined = torch.cat([only_tweet,du1_embedding, du2_embedding,before_after_embedding], dim=0)
            enc_logits.append(combined)
            #print('here')
        enc_logits=torch.stack(enc_logits)
        # print('enc_shape', enc_logits.size())
        output_logits = self.linear_classifier(enc_logits).to(device)

        loss =self.loss_fn(output_logits.to(device), labels.to(device))

        return {
            "loss": loss.to(device),
            "logits": output_logits.to(device),
            "labels":labels.to(device)
        }

    def data_collator(self, inst):

        #print(inst)
        diss_pairs_ind = []
        labels =torch.tensor([])
        tweets = []
        for i in inst:
            tweets.append(i['tweet'])
            labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)

        full_tweets_enc = self.enc_tok(tweets,
                                        return_tensors="pt", padding=True,
                                        truncation=True, max_length=512)
        #print(full_tweets_enc['input_ids'][0], inst[0])
        #print(full_tweets_enc['input_ids'][1],inst[1])

        sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]
        #print(full_tweets_enc)
        #print(sep_indices)
        diss_pairs_ind = [ ]
        for i in range(len(inst)):
            sep_indices_twt = sep_indices[i]
            #print(i,sep_indices_twt)
            #du_pairs = [0,1]#inst[i]['du_pairs_index']
            len_sep_indices_twt = len(sep_indices_twt)
            #print(du_pairs)

            tweet_start_end = [0,sep_indices_twt[0]]
            du1_start_end = [sep_indices_twt[0]+1, sep_indices_twt[1]]
            du2_start_end = [sep_indices_twt[1]+1, sep_indices_twt[2]]
            before_after_start_end = [sep_indices_twt[2]+1, sep_indices_twt[4]]
            # for p in du_pairs:
            #     temp = []
            #     if p ==0:
            #         temp.append(0)
            #         temp.append(sep_indices_twt[0])
            #     elif p==len_sep_indices_twt:
            #         temp.append(sep_indices_twt[-1]+1)
            #         temp.append(len(full_tweets_enc['input_ids'][i]))
            #     else:
            #         temp.append(sep_indices_twt[p-1]+1)
            #         temp.append(sep_indices_twt[p])
            #     pair_dus.append(temp)
            pair_dus = [tweet_start_end,du1_start_end,du2_start_end,before_after_start_end]
            diss_pairs_ind.append(pair_dus)
        #print(diss_pairs_ind[0])
        #print(diss_pairs_ind[1])
        return {"tweet": full_tweets_enc,
                "diss_pairs_ind": torch.tensor(diss_pairs_ind),
                "labels": labels.long()
                }


class DissonanceClassifierTweetandDus(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()
        # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
        self.num_labels = 3
        self.with_context = True
        self.dropout = 0.1

        self.loss_fn = torch.nn.CrossEntropyLoss()
        model_name = "roberta-base"
        self.enc_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels,
                                                                            hidden_dropout_prob=self.dropout)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 3, 500), torch.nn.ReLU(True),
                                                     torch.nn.Linear(500, self.num_labels))

    def forward(self, context, list_of_pairs_enc,tweet_du_ind_ranges, labels):

        n = len(list_of_pairs_enc)
        loss = torch.tensor(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.tensor([]).to(device)
        # print(list_of_pairs_enc)
        for i in range(n):
            tweets_embeddings = \
            self.enc_model(input_ids=list_of_pairs_enc[i]["input_ids"], attention_mask=list_of_pairs_enc[i]["attention_mask"], output_hidden_states=True).hidden_states[-1]
            n_pairs = len(list_of_pairs_enc[i]['input_ids'])
            # print(tweets_embeddings.size())
            # print(n_pairs)
            enc_logits=[]
            for j in range(n_pairs):
                one_tweet_enconding = tweets_embeddings[j]
                # print(one_tweet_enconding.size())
                #one_tweet_enconding
                # print(tweet_du_ind_ranges[i][j])
                only_tweet = torch.mean(one_tweet_enconding[tweet_du_ind_ranges[i][j][0][0]:tweet_du_ind_ranges[i][j][0][1]], dim=0)
                du1_embedding = torch.mean(one_tweet_enconding[tweet_du_ind_ranges[i][j][1][0]:tweet_du_ind_ranges[i][j][1][1]], dim=0)
                du2_embedding = torch.mean(one_tweet_enconding[tweet_du_ind_ranges[i][j][2][0]:tweet_du_ind_ranges[i][j][2][1]], dim=0)
                combined = torch.cat([only_tweet, du1_embedding, du2_embedding], dim=0)
                enc_logits.append(combined)

            # print(combined[0])
            enc_logits = torch.stack(enc_logits)
            output_logits = self.linear_classifier(enc_logits).to(device)

            loss = loss + self.loss_fn(output_logits, labels[i].to(device))
            # print(loss)
            logits = torch.cat((logits, output_logits), dim=0)

        return {
            "loss": loss,
            "logits": logits,
            "labels": torch.cat(labels, dim=0)
        }

    def data_collator(self, inst):
        # print(inst)

        # full_tweets = [i["tweet"].replace("<", "$123$").replace(" > ", "$321$") for i in inst] #" ".join([i.strip() for i in (" ".join([each.strip() for each in inst['tweet'].split("<")])).split(">")])
        full_tweets = []
        tweet_diss_pairs_enc = []
        tweet_du_ind_ranges=[]
        labels = []
        #print(inst)
        for i in inst:
            full_tweets.append(i['tweet'])

            full_text = i['list_pairs']

            full_tweets_enc = self.enc_tok(full_text,
                                                        return_tensors="pt", padding=True,
                                                        truncation=True, max_length=512)
            sep_indices = [[index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in
                           full_tweets_enc['input_ids']]
            sep_ind_ranges=[]
            for j in range(len(full_text)):
                sep_indices_twt = sep_indices[j]
                tweet_start_end = [0, sep_indices_twt[0]]
                du1_start_end = [sep_indices_twt[0] + 1, sep_indices_twt[1]]
                du2_start_end = [sep_indices_twt[1] + 1, sep_indices_twt[2]]
                sep_ind_ranges.append([tweet_start_end,du1_start_end,du2_start_end])
            tweet_du_ind_ranges.append(sep_ind_ranges)
            tweet_diss_pairs_enc.append(full_tweets_enc)
            #print(i['labels'])
            labels.append(torch.tensor(i['labels']))
        #print(full_tweets)
        #print(tweet_diss_pairs_enc,tweet_du_ind_ranges)
        return {"context": full_tweets,
                "list_of_pairs_enc": tweet_diss_pairs_enc,
                "tweet_du_ind_ranges": tweet_du_ind_ranges,
                "labels": labels}


# class pdtbClassifierFullTweetSep(torch.nn.Module):
#
#     def __init__(self, *args):
#         super().__init__()
#         # self.loss_func =  FocalLoss(gamma=5, alpha=0.25)
#         self.num_labels = 2
#         self.with_context = True
#         self.dropout = 0.1
#
#         self.loss_fn = torch.nn.CrossEntropyLoss()
#         model_name = "roberta-base"
#         self.enc_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=self.dropout)
#         self.enc_tok = AutoTokenizer.from_pretrained(model_name)
#         self.enc_tok.add_tokens(["$123$"])
#         self.enc_model.resize_token_embeddings(len(self.enc_tok))
#
#         self.linear_classifier = torch.nn.Sequential(torch.nn.Linear(768 * 2, 500), torch.nn.ReLU(True),
#                                                      torch.nn.Linear(500, self.num_labels))
#
#     def forward(self, tweet, diss_pairs_ind, labels):
#
#         tweets_embeddings = self.enc_model(input_ids=tweet["input_ids"],
#                                            attention_mask= tweet["attention_mask"], output_hidden_states=True).hidden_states[-1]
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # print(tweets_embeddings.shape)
#         n = len(diss_pairs_ind)
#
#         enc_logits = []  # .to(device)
#         for i in range(n):
#             one_tweet_enconding=tweets_embeddings[i]
#             #print(one_tweet_enconding.size())
#
#             du1_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][0][0]:diss_pairs_ind[i][0][1]], dim=0)
#             du2_embedding = torch.mean(one_tweet_enconding[diss_pairs_ind[i][1][0]:diss_pairs_ind[i][1][1]], dim=0)
#
#             #one_tweet_embedding = torch.mean(one_tweet_enconding,dim=0)
#             combined = torch.cat([du1_embedding, du2_embedding], dim=0)
#             enc_logits.append(combined)
#             #print('here')
#         enc_logits=torch.stack(enc_logits)
#         # print('enc_shape', enc_logits.size())
#         output_logits = self.linear_classifier(enc_logits).to(device)
#
#         loss =self.loss_fn(output_logits.to(device), labels.to(device))
#
#         return {
#             "loss": loss.to(device),
#             "logits": output_logits.to(device),
#             "labels":labels.to(device)
#         }
#
#     def data_collator(self, inst):
#
#         #print(inst)
#         diss_pairs_ind = []
#         labels =torch.tensor([])
#         tweets = []
#         for i in inst:
#             tweets.append(i['tweet'])
#             labels=torch.cat((labels,torch.tensor([i['labels']])), dim=0)
#
#         full_tweets_enc = self.enc_tok(tweets,
#                                         return_tensors="pt", padding=True,
#                                         truncation=True, max_length=512)
#         # print(full_tweets_enc['input_ids'][0], inst[0])
#         # print(full_tweets_enc['input_ids'][1],inst[1])
#
#         sep_indices = [ [index for index, num in enumerate(tweetenc) if num == 2] for tweetenc in full_tweets_enc['input_ids'] ]
#
#         diss_pairs_ind = [ ]
#         for i in range(len(inst)):
#             sep_indices_twt = sep_indices[i]
#             #print(i,sep_indices_twt)
#             du_pairs = [0,1]#inst[i]['du_pairs_index']
#             len_sep_indices_twt = len(sep_indices_twt)
#             #print(du_pairs)
#             pair_dus=[]
#             for p in du_pairs:
#                 temp = []
#                 if p ==0:
#                     temp.append(0)
#                     temp.append(sep_indices_twt[0])
#                 elif p==len_sep_indices_twt:
#                     temp.append(sep_indices_twt[-1]+1)
#                     temp.append(len(full_tweets_enc['input_ids'][i]))
#                 else:
#                     temp.append(sep_indices_twt[p-1]+1)
#                     temp.append(sep_indices_twt[p])
#                 pair_dus.append(temp)
#             diss_pairs_ind.append(pair_dus)
#         return {"tweet": full_tweets_enc,
#                 "diss_pairs_ind": torch.tensor(diss_pairs_ind),
#                 "labels": labels.long()
#                 }