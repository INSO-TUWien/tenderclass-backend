import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.classifier.pytorchModells.PyTorchTransformer import PyTorchTransformer
from src.entity.LabeledTenderCollection import LabelledTenderCollection

class FullTextBertModel(MlModelInterface):

    def __init__(self):
        # Load the BERT tokenizer
        self.tokenizerLong = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.modelLong = AutoModel.from_pretrained("allenai/longformer-base-4096")
        self.tokenizerShort = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.modelShort = AutoModel.from_pretrained("distilbert-base-uncased")
        self.max_len = 128
        self.batch_size = 32
        self.epochs = 4
        self.evaluation = True
        self.loss_fn = torch.nn.CrossEntropyLoss()


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def classify(self, tenders):
        pass

    def train(self, labelled_tenders):
        torch.cuda.empty_cache()
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        training_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(), "description": labelled_tenders_collection.get_descriptions(), "label": labelled_tenders_collection.get_labels()})

        X = training_df["title"].values
        y = training_df["label"].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        print("Tokenizing data...")
        train_inputs, train_masks = self.preprocessing_for_bert(X_train)
        val_inputs, val_masks = self.preprocessing_for_bert(X_val)

        # Convert other data types to torch.Tensor
        train_labels = torch.tensor(y_train)
        val_labels = torch.tensor(y_val)

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size)

        #train
        # Instantiate Bert Classifier
        bert_classifier = PyTorchTransformer(modelLong=self.modelLong, modelShort=self.modelShort, freeze_bert=True)

        # Tell PyTorch to run the model on GPU
        bert_classifier.to(self.device)

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * self.epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=total_steps)

        # Start training loop
        print("Start training...\n")
        for epoch_i in range(self.epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            bert_classifier.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                bert_classifier.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = bert_classifier(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(bert_classifier.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if self.evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate(bert_classifier, val_dataloader)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

        print("Training complete!")

    def evaluate(self, model, val_dataloader):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = pd.np.mean(val_loss)
        val_accuracy = pd.np.mean(val_accuracy)

        return val_loss, val_accuracy

    def create_new_model(self):
        pass

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for text in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizerShort.encode_plus(
                truncation=True,
                text=text,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks
