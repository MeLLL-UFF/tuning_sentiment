list_of_datasets=(HCR Narr-KDML-2012 STS-gold SemEval13 SemEval15-Task11 SemEval16 SemEval17-test SemEval18 SentiStrength
Target-dependent VADER archeage debate08 hobbit iphone irony movie ntua person sanders sarcasm sentiment140)

Models=(RoBERTa BERT  BERTweet)
seeds=(12 34 56)
OriginalModels=(roberta-base bert-base-uncased BERTweet_base_transformers)

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
      #NameTag=${models}${Dataset}"InData"${FinetuningStrategy} #tag dos arquivos e modelos salvos
      NameTag=${models}"_"${Dataset}"_"${seed}"_"${FinetuningStrategy} #tag dos arquivos e modelos salvos
      InputFine=${Dataset}"_evaluation.txt" #path to finetuning file
      InputAval=${Dataset}"_evaluation.csv" #path to evaluation step
      OriginalModel=${OriginalModels[i]} #path or name of original model: bert-base-uncased, roberta-base or BERTweet_base_transformers
      echo ${InputAval}


      ##Montando infra de pastas
      mkdir -p ./${NameTag} #caminho salvar o modelo tunado
      mkdir -p /sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar output da classificação

      ## Caminhos
      PathInputAval=/sentiment-embeddings/projeto-modularizar/inputs/${InputAval} #caminho do input do dataset para classificação
      PathInputFine=/sentiment-embeddings/projeto-modularizar/inputs/${InputFine} #caminho do input para fine tuning
      PathOutputDataset=/sentiment-embeddings/projeto-modularizar/outputs/${NameTag} #caminho para salvar resultado da classificação
      PathOutputModel=./${NameTag} # caminho do modelo treinado


      ## Executando Fine tuning
      python3 Fine_tuning_models.py -p ${OriginalModel} -i ${PathInputFine} -o ${PathOutputModel} -t ${FinetuningStrategy} -m ${embedding} -s ${seed}

      ## Executando Avaliação do modelo tunado
      python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${PathOutputModel} -e ${embedding} -p ${NameTag} -t ${OriginalModel} -s ${seed}
      i=$i+1

    done
  done
done
