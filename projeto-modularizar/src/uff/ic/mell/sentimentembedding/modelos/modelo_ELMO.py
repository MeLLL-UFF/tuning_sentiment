import tensorflow_hub as hub
#from uff.ic.mell.sentimentembedding.modelos.modelo import Modelo
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import pandas as pd

##########
# Source Allen Ai: https://allennlp.org/elmo
# tutorial tensorflow hub: https://medium.com/@prasad.pai/how-to-use-tensorflow-hub-with-code-examples-9100edec29af
# tutorial https://towardsdatascience.com/transfer-learning-using-elmo-embedding-c4a7e415103c
# tutorial II: https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440
#######

class ModeloELMO():
    def __init__(self,OutputType='default', train=False):
        """
        :param OutputType:
            word_emb: the character-based word representations with shape [batch_size, max_length, 512].
            lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
            lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
            elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
            default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
        """
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=train)
        self.name = "ModelELMO"
        self.outputtype = OutputType

    def get_tweet_embed(self, sentences,sess,**kwagars) -> pd.DataFrame:
        embeddings = self.elmo(sentences, signature="default", as_dict=True)[self.outputtype]
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        if (self.outputtype=='default'):
            return sess.run(embeddings)
        else:
            if(self.outputtype=='elmo'):
                return sess.run(tf.reduce_mean(embeddings, 1))

    def embTexts(self, dataSeries:pd.Series, **kwagars) -> pd.DataFrame:
        embeddings = np.ndarray((len(dataSeries), 1024))
        with tf.Session() as sess:
            embeddings = self.get_tweet_embed(dataSeries, sess)
            return pd.DataFrame(embeddings)

if __name__ == "__main__":
    elmo = ModeloELMO()
    df=pd.DataFrame()
    list_train = ['I love Columbia', 'I love rio']
    df=elmo.embTexts(list_train)
    print(df)
    print(np.sum(df.iloc[0]-df.iloc[1]))