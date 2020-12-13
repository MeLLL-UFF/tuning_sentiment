from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from uff.ic.mell.sentimentembedding.modelos.down_task_finetune import Fine_tune_Modelo
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments
from transformers import RobertaForSequenceClassification
from fairseq.data.encoders.fastbpe import fastBPE
import argparse

from transformers import RobertaConfig
from transformers import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from dataclasses import dataclass
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, Dict, List, NewType, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer



class BertweetTokenizer():
    def __init__(self,path,_pad_token=False):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes',
                            default=path + '/bpe.codes',
                            required=False,
                            type=str,
                            help='path to fastBPE BPE')
        args, unknown = parser.parse_known_args()
        self.bpe = fastBPE(args)
        self.vocab = Dictionary()
        self.vocab.add_from_file(path + "/dict.txt")
        self._pad_token = _pad_token
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.pad_token = '<pad>'
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        self.mask_token = "<mask>"


    def tokenize(self,text):
        subwords = self.bpe.encode(text)
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if self._pad_token==True:
            token_size = len(input_ids)
            if token_size<129:
                input_ids = input_ids + [1] * (128 - token_size)
            else:
                #dif = abs(130 - token_size)
                input_ids = input_ids[:128]
        return [0] + input_ids + [2]

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab.__len__() + 4

    def convert_tokens_to_ids(self, tokens):
        input_ids = self.vocab.encode_line(tokens, append_eos=False, add_if_not_exist=False).long().tolist()
        return torch.LongTensor(input_ids)

    def decode_id(self, id):
        return self.vocab.string(id, bpe_symbol='@@')

    def decode_id_nospace(self, id):
        return self.vocab.string(id, bpe_symbol='@@ ')

    def get_special_tokens_mask(self,ids,already_has_special_tokens=True ):
        mask=[]
        #print(ids)
        for id in ids:
            if id in [self.cls_token_id, self.sep_token_id, self.pad_token_id, self.mask_token_id]:
                mask.append(1)
            else:
                mask.append(0)
        #print(mask)
        return mask


class LineByLineTextDatasetBertweet(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    def __init__(self, tokenizer_path: str, file_path: str):
        super().__init__()

        tokenizer = BertweetTokenizer(tokenizer_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = list(map(tokenizer.tokenize,lines))

    def max_lenght(self):
        max=-9
        for sentences in self.examples:
            if len(sentences)>max:
                max=len(sentences)

        return  max
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class Modelobertweet(ModeloTransformer):
    def __init__(self, model_path:str,token_path:str,Method=None,train=False):
        self.name = model_path+"_"+Method
        self.method = Method
        self.path = str(model_path)
        self.config = RobertaConfig.from_pretrained(self.path+'/config.json')
        self.originalModel = RobertaModel.from_pretrained(self.path+'/pytorch_model.bin',config=self.config,)
        self.tokenizer=BertweetTokenizer(token_path,train)

    def getEmbeddings(self,tokens_ids):
        all_input_ids = torch.tensor([tokens_ids], dtype=torch.long).cuda()
        self.originalModel.cuda()
        with torch.no_grad():
            if self.method == "CONTEXT":
                features = self.originalModel(all_input_ids)
                hidden_states = features[2]
                last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
                # cast layers to a tuple and concatenate over the last dimension
                cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
                # take the mean of the concatenated vector over the token dimension
                cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1)
            elif self.method == "STATIC":
                features = self.originalModel.get_input_embeddings()(all_input_ids)
                cat_sentence_embedding = torch.mean(features[0], dim=0)
        return cat_sentence_embedding.squeeze().cpu().numpy()

    def embTexts(self,Series):
        tokensid = Series.apply(self.tokenizer.tokenize)
        df = pd.DataFrame(tokensid.apply(self.getEmbeddings))
        return pd.DataFrame(df.iloc[:,0].tolist(), index= df.iloc[:,0].index)

    def fine_tune_LM(self,input_text_path,output_path,pathOriginModel=None):
        model = RobertaForMaskedLM.from_pretrained(self.path)

        dataset = LineByLineTextDatasetBertweet(
            tokenizer_path=self.path,
            file_path=input_text_path
        )
        print('max length',dataset.max_lenght())
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=50_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            prediction_loss_only=True
        )

        print("Beginning Fine tuning")
        trainer.train()
        trainer.save_model(output_path)

    def fine_tune_downtask(self,dataset,epochs,batch,outputDir,learn_rate,weight_decay):
        print("Building Fine tune model object")
        fine_tune = Fine_tune_Modelo(RobertaForSequenceClassification,
                                     self.vocab,self.tokenizer,dataset,
                                     epochs,batch,outputDir,learn_rate,weight_decay)

        print("Tokenizing dataset")
        input_ids, attention_masks, labels = fine_tune.tokenizer_dataset()

        print("Spliting dataset")
        fine_tune.split_dataset(input_ids, attention_masks, labels,0.9)

        print("Building Dataloader")
        fine_tune.build_Dataloader()

        print("Begining Fine tuning")
        fine_tune.train_finetune()
