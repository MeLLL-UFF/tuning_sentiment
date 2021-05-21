from transformers import BertConfig, BertTokenizer, BertModel
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast

import numpy as np
import pandas as pd

class ModeloBert(ModeloTransformer):
    def __init__(self, vocabulary:str, tokenizer:str,embedMethod:ModeloTransformer.METHOD):
        super().__init__("Bert: "+vocabulary+" / "+ModeloTransformer.METHOD(embedMethod).name,
                        config= BertConfig(), 
                        tokenizer= BertTokenizer.from_pretrained(tokenizer),
                        originalModel= BertModel.from_pretrained(vocabulary, output_hidden_states=True),
                        embedMethod= embedMethod)

        self.vocab = vocabulary

    def fine_tune_LM(self,input_text_path,output_path,tokenizer_path=None):
        if tokenizer_path!=None:
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        else:
            tokenizer = BertTokenizerFast.from_pretrained(self.vocab)
        model = BertForMaskedLM.from_pretrained(self.vocab)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=input_text_path,
            block_size=512,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            prediction_loss_only=True,
        )
        trainer.train()
        trainer.save_model(output_path)
        return trainer.model

