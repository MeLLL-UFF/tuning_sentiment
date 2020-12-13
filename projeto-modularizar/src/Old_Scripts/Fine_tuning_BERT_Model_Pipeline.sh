
## Parâmetros
embedding='BERT' # RoBERTaouBERT
NameTag="BERTweet50000"
InputFine="eding_50000.txt"
InputAval="dataset_consolidado_ppbase_2.csv"
FinetuningStrategy="LM" # LM our Downtask
OriginModel='bert-base-uncased'


##Montando infra de pastas
mkdir -p ./${NameTag} #caminho salvar o modelo tunado
mkdir -p ./${NameTag}/Tokenizer #caminho salvar o tokenizer treinado
mkdir -p /sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar output da classificação

## Caminhos
PathInputAval=/sentiment-embeddings/projeto-modularizar/inputs/${InputAval} #caminho do input do dataset para classificação
PathInputFine=/sentiment-embeddings/projeto-modularizar/inputs/${InputFine} #caminho do input para fine tuning
PathOutputModel=./${NameTag} # caminho do modelo treinado
PathOutputDataset=/sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar resultado da classificação

## Executando Fine tuning
python3 Fine_tuning_models.py -p${OriginModel} -i ${PathInputFine} -o ${PathOutputModel} -t ${FinetuningStrategy} -m ${embedding}

## Executando Avaliação do modelo tunado
python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${PathOutputModel} -e ${embedding} -p ${NameTag} -t${OriginModel}
