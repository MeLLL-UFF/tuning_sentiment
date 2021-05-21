list_of_datasets=(HCR Narr-KDML-2012 STS-gold SemEval13 SemEval15-Task11 SemEval16 SemEval17-test SemEval18 SentiStrength
Target-dependent VADER archeage debate08 hobbit iphone irony movie ntua person sanders sarcasm sentiment140)

Models=(RoBERTa BERT  BERTweet)
seeds=(12 34 56)
OriginalModels=(roberta-base bert-base-uncased BERTweet_base_transformers)
evaltype='all_data_fn'
for seed in "${seeds[@]}"; do
  for Dataset in "${list_of_datasets[@]}"; do
    echo "Dataset: $Dataset"
    i=0
    for models in "${Models[@]}"; do
      echo "Model: $models"
      echo ${OriginalModels[i]}
      ## Parâmetros
      FinetuningStrategy="LM" # LM our Downtask
      embedding=${models} # RoBERTa ou BERT OU BERTweet OU Static
      dirBase="/Users/sergiojunior"
      NameTag=${models}"_"${Dataset}"_"${seed}"_"${FinetuningStrategy} #tag dos arquivos e modelos salvos
      InputFine=${Dataset}"_evaluation.txt" #path to finetuning file
      InputAval=${Dataset}"_evaluation.csv" #path to evaluation step
      OriginalModel=${OriginalModels[i]} #path or name of original model: bert-base-uncased, roberta-base or vinai/bertweet-base
      echo ${InputAval}


      ##Montando infra de pastas
      mkdir -p ./${NameTag} #caminho salvar o modelo tunado
      mkdir -p ${dirBase}/sentiment-embeddings/Project/outputs/${NameTag} #caminho para salvar output da classificação

      ## Caminhos
      PathInputAval=${dirBase}/sentiment-embeddings/Project/inputs/${InputAval} #caminho do input do dataset para classificação
      PathInputFine=${dirBase}/sentiment-embeddings/Project/inputs/${InputFine} #caminho do input para fine tuning
      PathOutputDataset=${dirBase}/sentiment-embeddings/Project/outputs/${NameTag} #caminho para salvar resultado da classificação
      PathOutputModel=./${NameTag} # caminho do modelo treinado

      ## Executando Avaliação do modelo tunado
      python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${PathOutputModel} \
                                -e ${OriginalModel} -p ${NameTag} -t ${OriginalModel} -s ${seed} -y ${evaltype}
      i=$i+1

    done
  done
done
