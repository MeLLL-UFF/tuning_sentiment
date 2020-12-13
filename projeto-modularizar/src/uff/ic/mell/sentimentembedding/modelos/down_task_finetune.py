from transformers import get_linear_schedule_with_warmup
import pandas as pd
from enum import Enum
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer
import torch
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(123)
from sklearn.utils import shuffle
'''
https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification
https://colab.research.google.com/drive/1PHv-IRLPCtv7oTcIGbsgZHqrB5LPvB7S#scrollTo=iANBiY3sLo-K
'''

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class Fine_tune_Modelo():
    def __init__(self, model_object, model_name,tokenizer,dataset,epochs,batch_size,outputDir, learn_rate,weight_decay):
        self.batch_size=batch_size
        self.outputDir = outputDir
        self.weight_decay = weight_decay
        self.train_dataloader=None
        self.eval_dataloader = None
        self.train_dataset = None
        self.eval_dataset = None
        self.tokenizer = tokenizer.from_pretrained(model_name, do_lower_case=True)
        df = pd.read_csv(dataset)
        df = shuffle(df)
        df_ = pd.DataFrame(df.iloc[:100000,:])
        df_ = df_.reset_index(drop=True)
        self.input_dataset =df_
        self.model = model_object.from_pretrained(model_name, # Use the 12-layer BERT model, with an uncased vocab.
                                                    num_labels = 2, # The number of output labels--2 for binary classification.
                                                                     # You can increase this for multi-class tasks.
                                                    output_attentions = False, # Whether the model returns attentions weights.
                                                    output_hidden_states = False, # Whether the model returns all hidden-states.
                                                    )

        self.epochs = epochs
        self.learn_rate = learn_rate
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.max_len = 280
    def tokenizer_dataset(self):
        input_ids = []
        attention_masks = []

        for sent in self.input_dataset['tweet']:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                max_length=self.max_len,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(self.input_dataset['classe'])

        return input_ids, attention_masks, labels


    def split_dataset(self,input_ids, attention_masks, labels,prop):
        # Combine the training inputs into a TensorDataset.
        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(prop * len(tensor_dataset))
        val_size = len(tensor_dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        self.train_dataset, self.eval_dataset = random_split(tensor_dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

    def build_Dataloader(self):


        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        self.train_dataloader = DataLoader(
            self.train_dataset,  # The training samples.
            shuffle=True,  # Select batches randomly
            batch_size=self.batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.eval_dataloader = DataLoader(
            self.eval_dataset,  # The validation samples.
            shuffle=True,  # Pull out batches sequentially.
            batch_size=self.batch_size  # Evaluate with this batch size.
        )

    def train_finetune(self):
        optimizer = AdamW(self.model.parameters(),
                          lr=self.learn_rate,
                          eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                          weight_decay=self.weight_decay,
                          )

        total_steps = len(self.train_dataloader)*self.epochs
        print('Total Steps: ',total_steps)
        print(len(self.train_dataloader))
        print(len(self.eval_dataloader))
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.1*total_steps,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        seed_val = 123

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []
        training_stats_steps = []
        evaluation_stats_steps = []


        # For each epoch...
        for epoch_i in range(0, self.epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')


            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()
            self.model.to(self.device)

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))
                    print('  Mean error train {}.'.format(total_train_loss/step))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()
                training_stats_steps.append(loss.item()/len(b_labels))
                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.eval_dataloader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = self.model(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()
                evaluation_stats_steps.append(loss.item())
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                accuracy = flat_accuracy(logits, label_ids)
                total_eval_accuracy += accuracy
                #print('  Accuracy validation {}.'.format(accuracy))

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.eval_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.eval_dataloader)


            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            #print("  Validation took: {:}".format(self.eval_dataloader))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy
                }
            )

        print("")
        print("Training complete!")

        self.model.save_pretrained(self.outputDir)


        df_training_stats = pd.DataFrame(training_stats)
        df_training_stats.to_csv(self.outputDir+'/training_stats.csv')
        plt.plot(df_training_stats['Training Loss'],'b-o', label='Train')
        plt.plot(df_training_stats['Valid. Loss'],'g-o',label='Validation')
        plt.xticks(list(df_training_stats['epoch']))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(self.outputDir+"/Training_and_Validation_Loss.png")
        plt.close()

        df_eval_stats_steps = pd.DataFrame(evaluation_stats_steps,columns=['Loss'])
        df_train_stats_steps = pd.DataFrame(training_stats_steps,columns=['Loss'])
        df_eval_stats_steps.to_csv(self.outputDir + '/eval_stats_steps.csv')
        df_train_stats_steps.to_csv(self.outputDir + '/training_stats_steps.csv')
        plt.plot(df_train_stats_steps['Loss'], 'b-o', label='Train')
        plt.plot(df_eval_stats_steps['Loss'], 'g-o', label='Validation')
        plt.xlabel("Steps")
        plt.legend()
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss by steps")
        plt.savefig(self.outputDir + "/Training_and_Validation_Loss_steps.png")
        