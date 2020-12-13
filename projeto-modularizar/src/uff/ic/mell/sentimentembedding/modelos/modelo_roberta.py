from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from uff.ic.mell.sentimentembedding.modelos.down_task_finetune import Fine_tune_Modelo
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification

class ModeloRoberta(ModeloTransformer):
    def __init__(self, vocabulary:str,tokenizer:str, embedMethod:ModeloTransformer.METHOD):
        super().__init__("Roberta: "+vocabulary+" / "+ModeloTransformer.METHOD(embedMethod).name,
                        config= RobertaConfig(),
                        tokenizer= RobertaTokenizer.from_pretrained(tokenizer),
                        originalModel= RobertaModel.from_pretrained(vocabulary, output_hidden_states=True),
                        embedMethod= embedMethod)

        self.vocab = vocabulary

    def fine_tune_LM(self,input_text_path,output_path,tokenizer_path):
        if tokenizer_path != None:
            tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained(self.vocab)
        model = RobertaForMaskedLM.from_pretrained(self.vocab)
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
            save_steps=50_000,
            save_total_limit=2,
        )
        print("Beginning Fine tuning")
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


if __name__ == "__main__":
    #import transformers
    #print(transformers.__version__)
    #roberta = ModeloRoberta('roberta-base','CONTEXT_CONCAT')
    #data_collator = DataCollatorForLanguageModeling(
    #    tokenizer=roberta.tokenizer, mlm=True, mlm_probability=0.15
    #)
    '''import pandas as pd
    import numpy as np
    df = pd.read_csv("/Usersimpo/sergiojunior/sentiment-embeddings/projeto-modularizar/inputs/Remoto/inputs/sentiment140_finetuning.csv")
    df.iloc[:,0]=df.iloc[:,0].apply(lambda x:x.strip())
    np.savetxt(r'./sentiment140_finetuning.txt', df.iloc[:,0], fmt='%s')

    f = open('./sentiment140_finetuning.txt', 'r')
    for lin in f.readlines()[:20]:
        print(lin)
        print("\n")
    f.close()'''