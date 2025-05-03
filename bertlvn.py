# This script is modified to REQUIRE CUDA and will exit if it's not available.

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel  # Use Auto classes
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter
import numpy as np
import copy  # For saving best model state
import time  # For timing
import sys  # For exiting the script

# --- Configuration ---
# Data Handling
CSV_FILE = r'C:\Users\collinguo\bertlegalvsnot\legal_text_classification.csv'  # Correct path here
TEXT_COLUMN = 'case_text'
TITLE_COLUMN = 'case_title'  # Added case title column
LABEL_COLUMN = 'case_outcome'
DATA_FRACTION = 1.0  # Use 1.0 for all data, 0.1 for 10%, etc. Set to 0.05 for quick testing

# Training Hyperparameters
LEARNING_RATE = 3.13e-5
BATCH_SIZE = 16
EPOCHS = 8  # Max number of epochs
DROPOUT_RATE = 0.16

# Early Stopping
EARLY_STOPPING_PATIENCE = 3  # Number of epochs to wait for improvement before stopping
EARLY_STOPPING_METRIC = 'val_f1_weighted'  # Metric to monitor ('val_loss' or 'val_f1_weighted')

# Loss Weighting Configuration
ENABLE_LOSS_WEIGHTS = True # Changed from UNDERSAMPLING to LOSS WEIGHTS
# TARGET_LABEL_ORDER is still used for ordering loss weights
TARGET_LABEL_ORDER = ['cited', 'referred to', 'applied', 'followed', 'considered', 'discussed', 'distinguished']
# Inverse proportions relative to the most frequent class, aligned with TARGET_LABEL_ORDER. Higher weight for less frequent class.
LOSS_WEIGHT_FACTORS = [3.04, 1.77, 1.36, 1.3, 1.14, 1.0, 1.0] # Example weights - ADJUST THESE BASED ON CLASS FREQUENCIES!!!

# Model & Tokenizer
MAX_LENGTH = 512  # Adjust as needed.


# --- Dataset Class ---
class LegalDataset(Dataset):
    def __init__(self, texts, titles, labels, tokenizer, max_length, use_title=False):
        self.texts = texts
        self.titles = titles  # added titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_title = use_title  # Add use_title flag

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if self.use_title:
            title = str(self.titles[idx])  # Retrieve title
            encoding_title = self.tokenizer.encode_plus(
                title,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return {
                'input_ids_text': encoding_text['input_ids'].flatten(),
                'attention_mask_text': encoding_text['attention_mask'].flatten(),
                'input_ids_title': encoding_title['input_ids'].flatten(),
                'attention_mask_title': encoding_title['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids_text': encoding_text['input_ids'].flatten(),
                'attention_mask_text': encoding_text['attention_mask'].flatten(),
                'input_ids_title': torch.tensor([]),  # Dummy tensor
                'attention_mask_title': torch.tensor([]),  # Dummy tensor
                'labels': torch.tensor(label, dtype=torch.long)
            }


# --- Model Definition ---
class LegalBertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels, hidden_size, dropout_rate=0.3, use_title=False):
        super(LegalBertClassifier, self).__init__()
        self.bert_text = AutoModel.from_pretrained(bert_model_name)
        self.use_title = use_title
        if use_title:
            self.bert_title = AutoModel.from_pretrained(bert_model_name)
            self.linear_text = nn.Linear(self.bert_text.config.hidden_size, hidden_size)
            self.linear_title = nn.Linear(self.bert_title.config.hidden_size, hidden_size)  # separate linear layer for title
            self.linear_combined = nn.Linear(2 * hidden_size, num_labels)  # combined hidden size
        else:
            self.linear_text = nn.Linear(self.bert_text.config.hidden_size, hidden_size)
            self.linear_combined = nn.Linear(hidden_size, num_labels)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_ids_text, attention_mask_text, input_ids_title, attention_mask_title):
        outputs_text = self.bert_text(input_ids=input_ids_text, attention_mask=attention_mask_text)
        pooled_output_text = torch.max(outputs_text.last_hidden_state, dim=1)[0]  # Max pooling

        x_text = self.dropout(pooled_output_text)
        x_text = self.linear_text(x_text)
        x_text = self.relu(x_text)

        if self.use_title:
            outputs_title = self.bert_title(input_ids=input_ids_title, attention_mask=attention_mask_title)
            pooled_output_title = torch.max(outputs_title.last_hidden_state, dim=1)[0]  # Max Pooling

            x_title = self.dropout(pooled_output_title)
            x_title = self.linear_title(x_title)
            x_title = self.relu(x_title)

            # Concatenate the outputs
            combined = torch.cat((x_text, x_title), dim=1)
        else:
            combined = x_text

        combined = self.dropout(combined)
        x = self.linear_combined(combined)

        return x


# --- Helper Function for Epoch Metrics ---
def calculate_metrics(labels, predictions, target_names):
    """Calculates and prints classification metrics."""
    print(classification_report(labels, predictions, target_names=target_names, zero_division=0))
    accuracy = accuracy_score(labels, predictions)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision_w:.4f}, Recall: {recall_w:.4f}, F1: {f1_w:.4f}")
    print(f"Macro Precision: {precision_m:.4f}, Recall: {recall_m:.4f}, F1: {f1_m:.4f}")
    return accuracy, f1_w  # Return weighted F1


# --- Main Execution ---
if __name__ == "__main__":
    # --- Device Selection (CUDA Check - MOVED TO TOP) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing device: {device}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # Print specific GPU name
    else:
        print("\nERROR: CUDA is not available. This script requires a CUDA-enabled GPU.")
        print("Exiting.")
        sys.exit(1)  # Exit the script if CUDA is not found


    # Hyperparameter sweep
    bert_model_names = ['colaguo/legalBERTclass8']  # Removed google-bert
    hidden_layer_sizes = [64, 128, 256]  # Removed 256, 512
    use_title_options = [True]  # use_title = False  - REMOVED IT, forcing True

    best_f1 = -1.0
    best_hyperparameters = None
    best_model_state = None

    #Adjust base_counts to match length of LOSS_WEIGHT_FACTORS array after drops
    # base_counts = [603, 1018, 1699,2252,2438,4363,12110] # Removed first 2 elements, then reversed
    # base_counts.reverse() # Unnecessary now

    for BERT_MODEL_NAME in bert_model_names:
        for hidden_size in hidden_layer_sizes:
            for use_title in use_title_options:  # Iterate through use_title options

                print("=" * 80)
                print(f"--- Starting Run with BERT Model: {BERT_MODEL_NAME}, Hidden Size: {hidden_size}, Use Title: {use_title} ---")
                print("=" * 80)

                print("--- Configuration ---")
                print(f"Using Model: {BERT_MODEL_NAME}")
                print(f"Hidden Layer Size: {hidden_size}")
                print(f"Use Title: {use_title}")
                print(f"Data Fraction: {DATA_FRACTION}")
                print(f"Max Sequence Length: {MAX_LENGTH}")
                print(f"Batch Size: {BATCH_SIZE}")
                print(f"Max Epochs: {EPOCHS}")
                print(f"Learning Rate: {LEARNING_RATE}")
                print(f"Dropout Rate: {DROPOUT_RATE}")
                print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
                print(f"Early Stopping Metric: {EARLY_STOPPING_METRIC}")
                print(f"Loss Weighting Enabled: {ENABLE_LOSS_WEIGHTS}") # UPDATED
                if ENABLE_LOSS_WEIGHTS: # UPDATED
                    print(f"Target Label Order for Loss Weights: {TARGET_LABEL_ORDER}") # UPDATED
                    print(f"Loss Weight Factors: {LOSS_WEIGHT_FACTORS}") # UPDATED
                print("-" * 20)

                # 1. Load Data
                print(f"\nLoading data from: {CSV_FILE}")
                try:
                    df = pd.read_csv(CSV_FILE)
                    print(f"Initial dataset size: {len(df)} rows")
                except FileNotFoundError:
                    print(f"Error: CSV file not found at {CSV_FILE}")
                    sys.exit(1)  # Exit if file not found

                # 2.  Drop rows BEFORE filtering for labels
                if use_title:
                    df = df.dropna(subset=[TEXT_COLUMN, TITLE_COLUMN, LABEL_COLUMN])  # Added title column
                else:
                    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])  # Drop titles only when necessary
                print(f"Dataset size after dropping NA: {len(df)} rows")

                # 2.  Filter out unwanted labels
                labels_to_remove = ['related', 'affirmed', 'approved']
                df = df[~df[LABEL_COLUMN].isin(labels_to_remove)]  # Filter out unwanted labels
                print(f"Dataset size after removing labels: {len(df)} rows")

                # 1.1 Apply Data Fraction
                if DATA_FRACTION < 1.0:
                    print(f"\nSampling {DATA_FRACTION * 100:.1f}% of the data...")
                    df = df.sample(frac=DATA_FRACTION, random_state=42).reset_index(drop=True)
                    print(f"Dataset size after sampling: {len(df)} rows")

                # 3. Preprocess Data
                print("\nPreprocessing data...")
                print("Null values before handling:")
                print(df.isnull().sum())  # This may now be redundant

                original_texts = df[TEXT_COLUMN].tolist()
                if use_title:
                    original_titles = df[TITLE_COLUMN].tolist()  # added title
                else:
                    original_titles = [''] * len(original_texts)  # Dummy titles
                original_labels = df[LABEL_COLUMN].tolist()

                unique_labels_in_data = sorted(list(set(original_labels)))
                label_map = {label: i for i, label in enumerate(unique_labels_in_data)}
                num_labels = len(label_map)
                target_names = [str(l) for l in unique_labels_in_data]

                print(f"\nFound {num_labels} unique labels in the data.")
                print(f"Label Mapping (Data Label -> Numerical Index): {label_map}")

                print("\nInitial Class Distribution (Original Data):")
                original_label_counts = Counter(original_labels)
                for label, count in sorted(original_label_counts.items()):
                    print(f"- {label}: {count} ({count / len(original_labels) * 100:.2f}%)")

                texts_to_use = original_texts
                titles_to_use = original_titles  # added title
                labels_to_use = [label_map[label] for label in original_labels]

                # --- Loss Weight Calculation ---
                if ENABLE_LOSS_WEIGHTS:
                    print("\nCalculating Loss Weights...")
                    inv_label_map = {v: k for k, v in label_map.items()}
                    class_weights = [0.0] * num_labels # Ensure size matches num_labels
                    weight_map = {}

                    # Initialise with ones to apply weights to classes with no defined target
                    for i in range(num_labels):
                        class_weights[i] = 1.0
                        weight_map[i] = 1.0

                    if len(LOSS_WEIGHT_FACTORS) != len(TARGET_LABEL_ORDER):
                        print(f"ERROR: LENGTH MISMATCH. len(LOSS_WEIGHT_FACTORS) = {len(LOSS_WEIGHT_FACTORS)} while len(TARGET_LABEL_ORDER) = {len(TARGET_LABEL_ORDER)}. Setting loss weighting to None and continuing.")
                        ENABLE_LOSS_WEIGHTS = False
                    else:
                        for i, target_label_name in enumerate(TARGET_LABEL_ORDER):
                            if target_label_name not in label_map:
                                print(f"Warning: Label '{target_label_name}' from TARGET_LABEL_ORDER not found in loaded data, skipping.")
                                continue

                            numerical_label_index = label_map[target_label_name]
                            weight_factor = LOSS_WEIGHT_FACTORS[i]
                            class_weights[numerical_label_index] = 1/weight_factor**1.2
                            weight_map[numerical_label_index] = 1/weight_factor**1.2

                        # Convert class weights to a tensor
                        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device) # ensure it's on GPU/CUDA

                        print("Calculated Class Weights:")
                        for i, weight in enumerate(class_weights):
                             label_name = inv_label_map.get(i, f"Unknown Index {i}")
                             print(f"- {label_name} (Index {i}): {weight:.3f}")

                # 4. Prepare Data for BERT
                print("\nPreparing data for BERT...")
                tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

                train_texts, val_texts, train_titles, val_titles, train_labels, val_labels = train_test_split(  # Added title splits
                    texts_to_use, titles_to_use, labels_to_use, test_size=0.2, random_state=42,
                    stratify=labels_to_use if len(set(labels_to_use)) > 1 else None
                )
                print(f"Training set size: {len(train_texts)}")
                print(f"Validation set size: {len(val_texts)}")

                train_dataset = LegalDataset(train_texts, train_titles, train_labels, tokenizer, MAX_LENGTH,
                                              use_title)  # Add titles to dataset
                val_dataset = LegalDataset(val_texts, val_titles, val_labels, tokenizer, MAX_LENGTH,
                                            use_title)  # Add titles to dataset

                train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

                # 5. Model Initialization & Device Selection (CUDA Check - MODIFIED)
                # --- REMOVE THIS ENTIRE BLOCK - ALREADY DEFINED
                # if torch.cuda.is_available():
                #     device = torch.device("cuda")
                #     print(f"\nUsing device: {device}")
                #     print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")  # Print specific GPU name
                # else:
                #     print("\nERROR: CUDA is not available. This script requires a CUDA-enabled GPU.")
                #     print("Exiting.")
                #     sys.exit(1)  # Exit the script if CUDA is not found
                # --- END REMOVE BLOCK ---

                model = LegalBertClassifier(BERT_MODEL_NAME, num_labels, hidden_size, DROPOUT_RATE,
                                             use_title).to(device)  # Move model to CUDA

                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

                # Instantiate CrossEntropyLoss with class weights
                if ENABLE_LOSS_WEIGHTS:
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()  # If not using weights, use default


                # --- Training Loop ---
                print("\n--- Starting Training ---")
                best_val_metric = float('inf') if EARLY_STOPPING_METRIC == 'val_loss' else float('-inf')
                epochs_no_improve = 0
                best_model_state_dict = None
                training_start_time = time.time()

                for epoch in range(EPOCHS):
                    epoch_start_time = time.time()
                    print("\n" + "=" * 60)
                    print(f"EPOCH {epoch + 1}/{EPOCHS}")
                    print("=" * 60)

                    # --- Training Phase ---
                    print("\n--- Training Phase ---")
                    model.train()
                    total_train_loss = 0
                    train_preds, train_labels_collected = [], []
                    batch_times = []

                    for batch_num, batch in enumerate(train_dataloader):
                        batch_start_time = time.time()
                        # MOVE DATA TO CUDA DEVICE
                        input_ids_text = batch['input_ids_text'].to(device)
                        attention_mask_text = batch['attention_mask_text'].to(device)
                        input_ids_title = batch['input_ids_title'].to(device)
                        attention_mask_title = batch['attention_mask_title'].to(device)
                        labels = batch['labels'].to(device)

                        optimizer.zero_grad()
                        outputs = model(input_ids_text, attention_mask_text, input_ids_title,
                                        attention_mask_title)  # Pass in title
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        batch_loss = loss.item()
                        total_train_loss += batch_loss

                        preds = torch.argmax(outputs, dim=1)
                        train_preds.extend(preds.cpu().numpy())
                        train_labels_collected.extend(labels.cpu().numpy())

                        batch_end_time = time.time()
                        batch_duration = batch_end_time - batch_start_time
                        batch_times.append(batch_duration)

                        if (batch_num + 1) % 20 == 0 or (batch_num + 1) == len(train_dataloader):
                            avg_batch_time = np.mean(batch_times[-20:])
                            print(f"  [Epoch {epoch + 1}] Batch {batch_num + 1}/{len(train_dataloader)} | "
                                  f"Loss: {batch_loss:.4f} | "
                                  f"Avg Batch Time (last 20): {avg_batch_time:.3f}s")

                    avg_train_loss = total_train_loss / len(train_dataloader)
                    print(f"\nEpoch {epoch + 1} Training Summary:")
                    print(f"Average Training Loss: {avg_train_loss:.4f}")
                    print("Training Metrics:")
                    _ = calculate_metrics(train_labels_collected, train_preds, target_names)

                    # --- Validation Phase ---
                    print("\n--- Validation Phase ---")
                    model.eval()
                    total_val_loss = 0
                    val_preds, val_labels_collected = [], []
                    val_start_time = time.time()

                    with torch.no_grad():
                        for batch_num, batch in enumerate(val_dataloader):
                            # MOVE DATA TO CUDA DEVICE
                            input_ids_text = batch['input_ids_text'].to(device)
                            attention_mask_text = batch['attention_mask_text'].to(device)
                            input_ids_title = batch['input_ids_title'].to(device)
                            attention_mask_title = batch['attention_mask_title'].to(device)
                            labels = batch['labels'].to(device)

                            outputs = model(input_ids_text, attention_mask_text, input_ids_title,
                                            attention_mask_title)  # Pass in title
                            loss = criterion(outputs, labels)
                            total_val_loss += loss.item()

                            preds = torch.argmax(outputs, dim=1)
                            val_preds.extend(preds.cpu().numpy())
                            val_labels_collected.extend(labels.cpu().numpy())

                    avg_val_loss = total_val_loss / len(val_dataloader)
                    val_end_time = time.time()
                    print(f"\nEpoch {epoch + 1} Validation Summary:")
                    print(f"Average Validation Loss: {avg_val_loss:.4f}")
                    print("Validation Metrics:")
                    val_accuracy, val_f1_weighted = calculate_metrics(val_labels_collected, val_preds, target_names)
                    print(f"Validation Phase Time: {val_end_time - val_start_time:.2f} seconds")

                    epoch_end_time = time.time()
                    print("-" * 40)
                    print(f"Epoch {epoch + 1} TOTAL TIME: {epoch_end_time - epoch_start_time:.2f} seconds")
                    print("-" * 40)

                    # --- Early Stopping Check ---
                    current_metric = avg_val_loss if EARLY_STOPPING_METRIC == 'val_loss' else val_f1_weighted
                    improved = False

                    if EARLY_STOPPING_METRIC == 'val_loss':
                        if current_metric < best_val_metric:
                            improved = True
                    else:  # Higher F1 is better
                        if current_metric > best_val_metric:
                            improved = True

                    if improved:
                        print(
                            f"*** Validation {EARLY_STOPPING_METRIC} improved ({best_val_metric:.4f} --> {current_metric:.4f}). Saving model... ***")
                        best_val_metric = current_metric
                        epochs_no_improve = 0
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                    else:
                        epochs_no_improve += 1
                        print(f"Validation {EARLY_STOPPING_METRIC} did not improve ({current_metric:.4f}).")
                        print(f"Epochs since last improvement: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print("\n!~!~! Early stopping triggered! !~!~!")
                        break

                training_end_time = time.time()
                print("\n--- Training Finished ---")
                print(f"Total Training Time: {training_end_time - training_start_time:.2f} seconds")

                # 6. Final Evaluation
                if best_model_state_dict:
                    print("\nLoading best model state for final evaluation...")
                    model.load_state_dict(best_model_state_dict)
                    model.to(device)  # Ensure model is on CUDA after loading
                else:
                    print("\nWarning: No best model state saved. Evaluating the last state.")
                    model.to(device)  # Ensure model is on CUDA

                print("\n--- Final Evaluation on Validation Set (Using Best Model) ---")
                model.eval()
                all_predictions = []
                all_labels = []
                eval_start_time = time.time()

                with torch.no_grad():
                    for batch in val_dataloader:
                        # MOVE DATA TO CUDA DEVICE
                        input_ids_text = batch['input_ids_text'].to(device)
                        attention_mask_text = batch['attention_mask_text'].to(device)
                        input_ids_title = batch['input_ids_title'].to(device)
                        attention_mask_title = batch['attention_mask_title'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(input_ids_text, attention_mask_text, input_ids_title,
                                        attention_mask_title)  # Pass in title
                        _, predictions = torch.max(outputs, 1)

                        all_predictions.extend(predictions.cpu().tolist())
                        all_labels.extend(labels.cpu().tolist())

                eval_end_time = time.time()
                print("Final Classification Report:")
                final_accuracy, final_f1_weighted = calculate_metrics(all_labels, all_predictions, target_names)
                print(f"Final Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")

                # Check if current model is the best
                if final_f1_weighted > best_f1:
                    best_f1 = final_f1_weighted
                    best_hyperparameters = {'bert_model': BERT_MODEL_NAME, 'hidden_size': hidden_size,
                                            'use_title': use_title}
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(
                        f"*** New best model found! F1: {best_f1:.4f} with hyperparameters: {best_hyperparameters} ***")

    # After the sweep, save the overall best model
    print("\n--- Hyperparameter Sweep Finished ---")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Hyperparameters: {best_hyperparameters}")

    if best_model_state is not None:
        use_title_str = "TextAndTitle" if best_hyperparameters['use_title'] else "TextOnly"
        SAVE_PATH = f"/{best_hyperparameters['bert_model'].replace('/', '_')}_best_model_loss_weighted_cuda_Hiddensize_{best_hyperparameters['hidden_size']}_F1_{best_f1:.4f}_{use_title_str}.pth"
        torch.save(best_model_state, SAVE_PATH)
        print(f"Best model saved to {SAVE_PATH}")
    else:
        print("No model was trained effectively during the sweep.")