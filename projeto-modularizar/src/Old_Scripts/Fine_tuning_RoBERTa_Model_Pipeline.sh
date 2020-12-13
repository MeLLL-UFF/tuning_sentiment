
## Parâmetros
FinetuningStrategy="LM" # LM our Downtask
embedding='RoBERTa' # RoBERTa ou BERT
NameTag="RoBERTXgboostFT1500"${FinetuningStrategy}
InputFine="sentiment140_finetuning.txt"
InputAval="dataset_consolidado_ppbase_2.csv"
OriginModel='roberta-base'


##Montando infra de pastas
mkdir -p ./${NameTag} #caminho salvar o modelo tunado
mkdir -p /sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar output da classificação

## Caminhos
PathInputAval=/sentiment-embeddings/projeto-modularizar/inputs/${InputAval} #caminho do input do dataset para classificação
PathInputFine=/sentiment-embeddings/projeto-modularizar/inputs/${InputFine} #caminho do input para fine tuning
PathOutputModel=./RoBER1500kLM # caminho do modelo treinado
PathOutputDataset=/sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar resultado da classificação

## Executando Fine tuning Language Model
#python3 Fine_tuning_models.py -p${OriginModel} -i ${PathInputFine} -o ${PathOutputModel} -t ${FinetuningStrategy} -m ${embedding}

## Executando Avaliação do modelo tunado
python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${PathOutputModel} -e ${embedding} -p ${NameTag} -t${OriginModel}
