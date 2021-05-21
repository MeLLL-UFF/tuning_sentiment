## Parâmetros
FinetuningStrategy="LM" # LM our Downtask
embedding='RoBERTa' # RoBERTa ou BERT OU BERTweet OU Static
NameTag="RoBERTa"${FinetuningStrategy} #tag dos arquivos e modelos salvos
seed=1234
dirBase="/Users/sergiojunior"
evaltype='normal'

InputAval="dataset_consolidado_ppbase_2.csv" #path to evaluation step
OriginalModel='roberta-base' #path or name of original model: bert-base-uncased, roberta-base or vinai/bertweet-base
ModelPath=${OriginalModel} #path of tuned model:/path or original model: OriginalModel

##Montando infra de pastas
mkdir -p ${dirBase}/sentiment-embeddings/Project/outputs/${NameTag} #caminho para salvar output da classificação

## Caminhos
PathInputAval=${dirBase}/sentiment-embeddings/Project/inputs/${InputAval} #caminho do input do dataset para classificação
PathOutputDataset=${dirBase}/sentiment-embeddings/Project/outputs/${NameTag} #caminho para salvar resultado da classificação

## Executando Avaliação do modelo tunado
python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${ModelPath} \
                            -e ${OriginalModel} -p ${NameTag} -t ${OriginalModel} -s ${seed} -y ${evaltype}