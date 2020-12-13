## Parâmetros
FinetuningStrategy="LM" # LM our Downtask
embedding='Static' # RoBERTa ou BERT OU BERTweet OU Static
NameTag="BERTweetNormalized"${FinetuningStrategy} #tag dos arquivos e modelos salvos
seed=123
dirBase="/estudo_orientado/"

InputAval="dataset_consolidado_ppbase_2.csv" #path to evaluation step
OriginalModel='BERTweet_base_transformers' #path or name of original model: bert-base-uncased, roberta-base or BERTweet_base_transformers
ModelPath=${OriginalModel} #path of tuned model:/path or original model: OriginalModel

##Montando infra de pastas
mkdir -p ${dirBase}/sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar output da classificação

## Caminhos
PathInputAval=${dirBase}/sentiment-embeddings/projeto-modularizar/inputs/${InputAval} #caminho do input do dataset para classificação
PathOutputDataset=${dirBase}/sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar resultado da classificação

## Executando Avaliação do modelo tunado
python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${ModelPath} -e ${embedding} -p ${NameTag} -t ${OriginalModel} -s ${seed}