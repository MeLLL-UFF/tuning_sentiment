
## Parâmetros
FinetuningStrategy="LM" # LM our Downtask
embedding='BERTweet' # RoBERTa ou BERT OU BERTweet OU Static
NameTag="BERTweetXgboost"${FinetuningStrategy} #tag dos arquivos e modelos salvos
InputFine="eding_50000.txt" #path to finetuning file
InputAval="dataset_consolidado_ppbase_2.csv" #path to evaluation step
OriginalModel='BERTweet_base_transformers' #path or name of original model: bert-base-uncased, roberta-base or BERTweet_base_transformers



##Montando infra de pastas
mkdir -p ./${NameTag} #caminho salvar o modelo tunado
mkdir -p /sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar output da classificação

## Caminhos
PathInputAval=/sentiment-embeddings/projeto-modularizar/inputs/${InputAval} #caminho do input do dataset para classificação
PathInputFine=/sentiment-embeddings/projeto-modularizar/inputs/${InputFine} #caminho do input para fine tuning
PathOutputDataset=/sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar resultado da classificação
PathOutputModel=./${NameTag} # caminho do modelo treinado


## Executando Fine tuning
#python3 Fine_tuning_models.py -p ${OriginalModel} -i ${PathInputFine} -o ${PathOutputModel} -t ${FinetuningStrategy} -m ${embedding}

## Executando Avaliação do modelo tunado
python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${OriginalModel} -e ${embedding} -p ${NameTag} -t ${OriginalModel}
