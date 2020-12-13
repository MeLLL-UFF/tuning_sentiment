
import pandas as pd
from sklearn.utils import shuffle
import argparse
import numpy as np
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str) 
    args = parser.parse_args()
    try:
      input_path = args.input_path
      df = pd.read_csv(input_path)
      path=os.path.dirname(input_path)
      train=0.60
      val=0.20
      test=0.20
      df = shuffle(df)
      df_train, df_val, df_test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
      df_train.to_csv(path+"/raw_dataset_finetuning/dataset.train.csv", index=False)
      df_val.to_csv(path+"/raw_dataset_finetuning/dataset.val.csv", index=False)
      df_test.to_csv(path+"/raw_dataset_finetuning/dataset.test.csv", index=False)

    except Exception as e:
        print(e)
        raise
