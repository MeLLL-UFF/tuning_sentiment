from uff.ic.mell.sentimentembedding.modelos.modelo_transformer import ModeloTransformer
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModel, AutoTokenizer 

class Modelobertweet(ModeloTransformer):
    def __init__(self, model_path:str,token_path:str,Method=None):
        super().__init__("Bertweet: "+model_path+" / "+ModeloTransformer.METHOD(Method).name,
                        config= '',
                        tokenizer= AutoTokenizer.from_pretrained(token_path),
                        originalModel= AutoModel.from_pretrained(model_path, output_hidden_states=False),
                        embedMethod= Method)

    def fine_tune_LM(self,input_text_path,output_path):
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