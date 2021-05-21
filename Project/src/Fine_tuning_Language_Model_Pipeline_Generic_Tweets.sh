seeds=(12 34 56)
edin=(25 50 250 500 1500 6600)
Models=(RoBERTa BERT BERTweet)
OriginalModels=(roberta-base bert-base-uncased BERTweet_base_transformers)
evaltype="normal"
for seed in "${seeds[@]}"; do
  i=0
  for models in "${Models[@]}"; do
    for edin_data in "${edin[@]}"; do
      echo "Model: $models"
      echo ${OriginalModels[i]}
      echo "seed: $seed"

      ## Parâmetros
      FinetuningStrategy="LM" # LM our Downtask
      embedding=${models} # RoBERTa ou BERT OU BERTweet OU Static
      NameTag=${models}"_"${edin_data}"_"${seed}"_"${FinetuningStrategy} #tag dos arquivos e modelos salvos
      InputFine="eding_"${edin_data}".txt" #path to finetuning file
      InputAval="dataset_consolidado_ppbase_2.csv" #path to evaluation step
      #OriginalModel='BERTweet_base_transformers' #path or name of original model: bert-base-uncased, roberta-base or vinai/bertweet-base
      OriginalModel=${OriginalModels[i]} #path or name of original model: bert-base-uncased, roberta-base or vinai/bertweet-base
      dirBase="/Users/sergiojunior"
      ##Montando infra de pastas
      mkdir -p ./${NameTag} #caminho salvar o modelo tunado
      mkdir -p ${dirBase}/sentiment-embeddings/Project/outputs/${NameTag} #caminho para salvar output da classificação

      ## Caminhos
      PathInputAval=${dirBase}/sentiment-embeddings/Project/inputs/${InputAval} #caminho do input do dataset para classificação
      PathInputFine=${dirBase}/sentiment-embeddings/Project/inputs/${InputFine} #caminho do input para fine tuning
      PathOutputDataset=${dirBase}/sentiment-embeddings/Project/outputs/${NameTag} #caminho para salvar resultado da classificação
      PathOutputModel=./${NameTag} # caminho do modelo treinado


      ## Executando Fine tuning
      python3 Fine_tuning_models.py -p ${OriginalModel} -i ${PathInputFine} -o ${PathOutputModel} -t ${FinetuningStrategy} -m ${embedding} -s ${seed}

      ## Executando Avaliação do modelo tunado
      python3 AvaliaModelos.py -i ${PathInputAval} -o ${PathOutputDataset} -m ${PathOutputModel} \
                                    -e ${OriginalModel} -p ${NameTag} -t ${OriginalModel} -s ${seed} -y ${evaltype}
      i=i+1
    done
  done
done