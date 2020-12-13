# Atualização Docker
```bash
apt install cuda
apt install git
apt install curl
apt install vim
apt install conda
```

# Conda
Trabalhar no ambiente UFF_sent-emb
```bash
conda activate UFF_sent-emb
```

# Python
python 3.8.2

# Instalações necessárias para python

```bash
conda install nltk
pip install emoji
conda install pandas
conda install keras
#pip install pytorch_transformers (subsituido pelo transformers)
#pip install transformers (substituido para pegar direto do git)
pip install git+https://github.com/huggingface/transformers
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install sklearn
pip install ekphrasis
pip install -U gensim
pip install fairseq
pip install fastBPE
pip install xgboost
pip install tensorflow_hub
conda install keras
```

# Comandos do docker da UFF

* Para logar na UFF
```bash
ssh <usuario>@200.20.15.153
```

* Para iniciar um docker com uma pasta mapeada(Este comando retorna um hash identificador do processo)
```bash
docker container run -itd --name ricsouzamoura --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=7, --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ricardo_moura/estudo_orientado:/estudo_orientado nvidia/cuda:9.0-devel_sent-emb_ricardomoura bash
```

* Para localizar o processo do docker
```bash
docker ps | grep <inicio do hash do processo>
```

* Para acessar o docker em execução (dar 2 enter para entrar)
```bash
docker attach <hash do processo>
```

* Para sair do terminal do container deixando ele rodando
```bash
 "Ctrl+P" "Ctrl+Q"
 ```

* Para sair do terminal do container finalizando ele
```bash
exit
```

* Para criar uma nova imagem (precisa rodar com o docker em execucao). Para pegar o nome da imagem em execução deve rodar docker ps
```bash
docker commit <nome da imagem em memoria(em execucao)> nvidia/cuda:9.0-devel_sent-emb_ricardomoura
```

* Para criar um conteiner a partir de uma imagem (imagem utilizada para finetuning: c4e58f48a70b)

```bash
docker run -it -v /home/sergio_barreto/sentiment-embeddings:/sentiment-embeddings c4e58f48a70b
```

# SCP para transferencia de arquivos entre as máquinas

* Da máquina local para a máquina remota:
copia README.md para a pasta teste (precisa colocar /home/<usuario> senao dá erro de permissão)
```bash
scp README.md <usuario>@<ip>:/home/<usuario>/teste
```

* Da máquina local para a máquina remota
copia um diretorio inteiro (precisa colocar /home/<usuario> senao dá erro de permissão)
```bash
scp -r /local/directory <usuario>@<ip>:/home/<usuario>/directory
```

* Da máquina remota para a máquina local 
copia o arquivo abc.txt da pasta teste para uma pasta. Neste exemplo usou-se o "." (pasta atual)
```bash
scp <usuario>@<ip>:/home/<usuario>/teste/abc.tt .
````

* Da máquina remota para a máquina local 
copia a pasta teste para uma pasta local.
```bash
scp -r <usuario>@<ip>:/home/<usuario>/teste /home/user/Desktop/
```

# RSYNC para transferencia de deltas

* da máquina local para a remota
```bash
rsync -a -P /opt/media/ remote_user@remote_host_or_ip:/opt/media/
```

* da máquina remota para a local
```bash
rsync -a -P remote_user@remote_host_or_ip:/opt/media/ /opt/media/
```


# Referências

* https://github.com/huggingface/transformers
* https://pytorch.org/docs/stable/index.html
* http://dockerlabs.collabnix.com/docker/cheatsheet/
* https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
