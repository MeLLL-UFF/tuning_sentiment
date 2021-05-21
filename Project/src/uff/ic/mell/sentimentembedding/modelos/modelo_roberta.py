from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizer

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

