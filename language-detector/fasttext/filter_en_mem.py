
indexes = []
#este arquivo de index foi pego com grep, então começa por 1
with open('fasttext__ld_index') as f:
    for line in f:
        #precisa reduzir um devido a forma como o grep recupera o numero da linha
        indexes.append(int(line)-1)

#a_file = open('teste.txt')
a_file = open('../twitterStream-20091110-20100201-v0.1.1')
newfile = []
#a leitura do arquivo começa da linha 0
for position, line in enumerate(a_file):
    if position in indexes:
        #print("linha> "+str(position))
        newfile.append(line)

with open('twitter_en_out.txt','w') as f:
  f.write('\n'.join(newfile))

a_file.close()
