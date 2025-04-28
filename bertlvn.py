import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel # Use Auto classes
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter
import numpy as np
import copy # For saving best model state
import time # For timing
import sys # For exiting the script
import optuna # For hyperparameter tuning


# --- Configuration ---
# Data Handling
CSV_FILE = "legal_text_classification.csv" # Correct path here
TEXT_COLUMN = 'case_text'
LABEL_COLUMN = 'case_outcome'
DATA_FRACTION = 1.0 # Use 1.0 for all data, 0.1 for 10%, etc. Set to 0.05 for quick testing

# Model & Tokenizer
BERT_MODEL_NAME = 'colaguo/legalclassBERTlarge' # Specific model from Hugging Face Hub
MAX_LENGTH = 512  # Adjust as needed.

# Training Hyperparameters (initial values - Optuna will tune)
LEARNING_RATE = 8e-5
#BATCH_SIZE = 16 # Reduced batch size might be needed for larger models like 'large' - Tuned
EPOCHS = 30 # Max number of epochs
#DROPOUT_RATE = 0.3 # Tuned

# Early Stopping
EARLY_STOPPING_PATIENCE = 5 # Number of epochs to wait for improvement before stopping
EARLY_STOPPING_METRIC = 'val_f1_weighted' # Metric to monitor ('val_loss' or 'val_f1_weighted')

# Undersampling Configuration
ENABLE_UNDERSAMPLING = True
TARGET_LABEL_ORDER = ['cited', 'referred to', 'applied', 'followed', 'considered', 'discussed', 'distinguished', 'related', 'affirmed', 'approved']
# Proportions relative to the most frequent class, aligned with TARGET_LABEL_ORDER.
UNDERSAMPLING_ARRAY = [.329**.5, .550**.5, .735**.5, .766**.5, .879**.5, 1, 1, 1, 1, 1] # Ensure length matches TARGET_LABEL_ORDER


# Optuna Setup
N_TRIALS = 10 # Number of Optuna trials
def objective(trial):
    # Define the hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    neuron_count_1 = trial.suggest_categorical("neuron_count_1", [64, 128, 256])
    neuron_count_2 = trial.suggest_categorical("neuron_count_2", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)


    # --- Dataset Class ---
    class LegalDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # --- Model Definition ---
    class LegalBertClassifier(nn.Module):
        def __init__(self, bert_model_name, num_labels, dropout_rate, neuron_count_1, neuron_count_2):
            super(LegalBertClassifier, self).__init__()
            self.bert = AutoModel.from_pretrained(bert_model_name)
            self.dropout = nn.Dropout(dropout_rate)
            self.linear1 = nn.Linear(self.bert.config.hidden_size, neuron_count_1)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(neuron_count_1, neuron_count_2)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(neuron_count_2, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = torch.max(outputs.last_hidden_state, dim=1)[0] # Max pooling

            x = self.dropout(pooled_output)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.relu2(x)
            x = self.dropout(x)
            x = self.linear3(x)

            return x

    # --- Helper Function for Epoch Metrics ---
    def calculate_metrics(labels, predictions, target_names):
        """Calculates and prints classification metrics."""
        #print(classification_report(labels, predictions, target_names=target_names, zero_division=0)) # Disable detailed report during tuning
        accuracy = accuracy_score(labels, predictions)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        #print(f"Overall Accuracy: {accuracy:.4f}") # Disable printing during tuning
        #print(f"Weighted Precision: {precision_w:.4f}, Recall: {recall_w:.4f}, F1: {f1_w:.4f}") # Disable printing during tuning
        #print(f"Macro Precision: {precision_m:.4f}, Recall: {recall_m:.4f}, F1: {f1_m:.4f}") # Disable printing during tuning
        return accuracy, f1_w # Return weighted F1


    # 1. Load Data
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    if DATA_FRACTION < 1.0:
        df = df.sample(frac=DATA_FRACTION, random_state=42).reset_index(drop=True)

    original_texts = df[TEXT_COLUMN].tolist()
    original_labels = df[LABEL_COLUMN].tolist()

    unique_labels_in_data = sorted(list(set(original_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels_in_data)}
    num_labels = len(label_map)
    target_names = [str(l) for l in unique_labels_in_data]


    # Split data BEFORE undersampling
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        original_texts, original_labels, test_size=0.2, random_state=42,
        stratify=original_labels if len(set(original_labels)) > 1 else None
    )

    # Map labels to numerical indices *after* the split.
    train_labels_numerical = [label_map[label] for label in train_labels]
    val_labels_numerical = [label_map[label] for label in val_labels]

    # 2.1 Apply Undersampling (if enabled) *ONLY* to the TRAINING data
    if ENABLE_UNDERSAMPLING:
        train_label_counts = Counter(train_labels) # Use train_labels here

        if not train_label_counts:
             print("Warning: No labels found in the TRAINING data. Cannot perform undersampling.")
             ENABLE_UNDERSAMPLING = False
        else:
            most_frequent_label_name = max(train_label_counts, key=train_label_counts.get)
            max_count = train_label_counts[most_frequent_label_name]

            inv_label_map = {v: k for k, v in label_map.items()}
            target_counts = {}
            numerical_label_counts = Counter(train_labels_numerical)


            for i, target_label_name in enumerate(TARGET_LABEL_ORDER):
                if target_label_name not in label_map:
                    continue

                numerical_label_index = label_map[target_label_name]
                target_proportion = UNDERSAMPLING_ARRAY[i]
                target_count = int(target_proportion * max_count)
                current_count = numerical_label_counts.get(numerical_label_index, 0)
                final_target_count = min(target_count, current_count)
                target_counts[numerical_label_index] = final_target_count

            temp_df = pd.DataFrame({'text': train_texts, 'label': train_labels_numerical})
            indices_to_keep = []

            for numerical_label_index, target_size in target_counts.items():
                 label_indices = temp_df.index[temp_df['label'] == numerical_label_index].tolist()
                 if len(label_indices) > target_size:
                     sampled_indices = np.random.choice(label_indices, size=target_size, replace=False)
                     indices_to_keep.extend(sampled_indices)
                 else:
                      indices_to_keep.extend(label_indices)

            undersampled_df = temp_df.loc[indices_to_keep].copy()
            undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
            train_texts_undersampled = undersampled_df['text'].tolist()
            train_labels_numerical_undersampled = undersampled_df['label'].tolist()

        # Use the undersampled training data
        train_texts = train_texts_undersampled
        train_labels_numerical = train_labels_numerical_undersampled

    else:
        train_texts = train_texts
        train_labels_numerical = train_labels_numerical


    # 3. Prepare Data for BERT
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = LegalDataset(train_texts, train_labels_numerical, tokenizer, MAX_LENGTH)
    val_dataset = LegalDataset(val_texts, val_labels_numerical, tokenizer, MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # 4. Model Initialization & Device Selection (CUDA Check - MODIFIED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("\nERROR: CUDA is not available. This script requires a CUDA-enabled GPU.")
        print("Exiting.")
        sys.exit(1) # Exit the script if CUDA is not found

    model = LegalBertClassifier(BERT_MODEL_NAME, num_labels, dropout_rate, neuron_count_1, neuron_count_2).to(device) # Pass neuron counts to model


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # Use the tuned learning rate
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    best_val_metric = float('inf') if EARLY_STOPPING_METRIC == 'val_loss' else float('-inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_preds, train_labels_collected = [], []

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_collected.extend(labels.cpu().numpy())



        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_preds, val_labels_collected = [], []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_collected.extend(labels.cpu().numpy())


        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy, val_f1_weighted = calculate_metrics(val_labels_collected, val_preds, target_names)


        # --- Early Stopping Check ---
        current_metric = avg_val_loss if EARLY_STOPPING_METRIC == 'val_loss' else val_f1_weighted

        if EARLY_STOPPING_METRIC == 'val_loss':
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                epochs_no_improve = 0 # Reset counter
            else:
                epochs_no_improve += 1
        else: # Higher F1 is better
             if current_metric > best_val_metric:
                 best_val_metric = current_metric
                 epochs_no_improve = 0 # Reset counter
             else:
                 epochs_no_improve += 1


        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

        trial.report(current_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_metric # Return the monitored metric


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Optuna Hyperparameter Tuning ---")
    study = optuna.create_study(direction="minimize" if EARLY_STOPPING_METRIC == 'val_loss' else "maximize") #minimize for loss, maximize for F1
    study.optimize(objective, n_trials=N_TRIALS)

    print("--- Optuna Hyperparameter Tuning Finished ---")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # After tuning, you can retrain your model with the best hyperparameters
    # For example, retrieve the best parameters:
    best_batch_size = trial.params['batch_size']
    best_dropout_rate = trial.params['dropout_rate']
    best_neuron_count_1 = trial.params['neuron_count_1']
    best_neuron_count_2 = trial.params['neuron_count_2']
    best_learning_rate = trial.params['learning_rate']


    # And then use these parameters to train your final model (replace the tuning loop with a full training loop):
    print("\n--- Training Final Model with Best Hyperparameters ---")

    # Load data again (or reuse existing data if it's still in memory)
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    if DATA_FRACTION < 1.0:
        df = df.sample(frac=DATA_FRACTION, random_state=42).reset_index(drop=True)

    original_texts = df[TEXT_COLUMN].tolist()
    original_labels = df[LABEL_COLUMN].tolist()

    unique_labels_in_data = sorted(list(set(original_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels_in_data)}
    num_labels = len(label_map)
    target_names = [str(l) for l in unique_labels_in_data]

    # Split data BEFORE undersampling
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        original_texts, original_labels, test_size=0.2, random_state=42,
        stratify=original_labels if len(set(original_labels)) > 1 else None
    )

    # Map labels to numerical indices *after* the split.
    train_labels_numerical = [label_map[label] for label in train_labels]
    val_labels_numerical = [label_map[label] for label in val_labels]

    # 2.1 Apply Undersampling (if enabled) *ONLY* to the TRAINING data
    if ENABLE_UNDERSAMPLING:
        train_label_counts = Counter(train_labels) # Use train_labels here

        if not train_label_counts:
             print("Warning: No labels found in the TRAINING data. Cannot perform undersampling.")
             ENABLE_UNDERSAMPLING = False
        else:
            most_frequent_label_name = max(train_label_counts, key=train_label_counts.get)
            max_count = train_label_counts[most_frequent_label_name]

            inv_label_map = {v: k for k, v in label_map.items()}
            target_counts = {}
            numerical_label_counts = Counter(train_labels_numerical)

            for i, target_label_name in enumerate(TARGET_LABEL_ORDER):
                if target_label_name not in label_map:
                    continue

                numerical_label_index = label_map[target_label_name]
                target_proportion = UNDERSAMPLING_ARRAY[i]
                target_count = int(target_proportion * max_count)
                current_count = numerical_label_counts.get(numerical_label_index, 0)
                final_target_count = min(target_count, current_count)
                target_counts[numerical_label_index] = final_target_count

            temp_df = pd.DataFrame({'text': train_texts, 'label': train_labels_numerical})
            indices_to_keep = []

            for numerical_label_index, target_size in target_counts.items():
                 label_indices = temp_df.index[temp_df['label'] == numerical_label_index].tolist()
                 if len(label_indices) > target_size:
                     sampled_indices = np.random.choice(label_indices, size=target_size, replace=False)
                     indices_to_keep.extend(sampled_indices)
                 else:
                      indices_to_keep.extend(label_indices)

            undersampled_df = temp_df.loc[indices_to_keep].copy()
            undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
            train_texts_undersampled = undersampled_df['text'].tolist()
            train_labels_numerical_undersampled = undersampled_df['label'].tolist()

        # Use the undersampled training data
        train_texts = train_texts_undersampled
        train_labels_numerical = train_labels_numerical_undersampled

    else:
        train_texts = train_texts
        train_labels_numerical = train_labels_numerical

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = LegalDataset(train_texts, train_labels_numerical, tokenizer, MAX_LENGTH)
    val_dataset = LegalDataset(val_texts, val_labels_numerical, tokenizer, MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=best_batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("\nERROR: CUDA is not available. This script requires a CUDA-enabled GPU.")
        print("Exiting.")
        sys.exit(1) # Exit the script if CUDA is not found

    final_model = LegalBertClassifier(BERT_MODEL_NAME, num_labels, best_dropout_rate, best_neuron_count_1, best_neuron_count_2).to(device) # Use best params
    optimizer = optim.AdamW(final_model.parameters(), lr=best_learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Full training loop with early stopping (similar to original code)
    best_val_metric = float('inf') if EARLY_STOPPING_METRIC == 'val_loss' else float('-inf')
    epochs_no_improve = 0
    best_model_state_dict = None
    training_start_time = time.time()

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
        return accuracy, f1_w # Return weighted F1


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print("\n" + "="*60)
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print("="*60)

        # --- Training Phase ---
        print("\n--- Training Phase ---")
        final_model.train()
        total_train_loss = 0
        train_preds, train_labels_collected = [], []
        batch_times = []

        for batch_num, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            # MOVE DATA TO CUDA DEVICE
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = final_model(input_ids, attention_mask)
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
                 print(f"  [Epoch {epoch+1}] Batch {batch_num+1}/{len(train_dataloader)} | "
                       f"Loss: {batch_loss:.4f} | "
                       f"Avg Batch Time (last 20): {avg_batch_time:.3f}s")


        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} Training Summary:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print("Training Metrics:")
        _ = calculate_metrics(train_labels_collected, train_preds, target_names)

        # --- Validation Phase ---
        print("\n--- Validation Phase ---")
        final_model.eval()
        total_val_loss = 0
        val_preds, val_labels_collected = [], []
        val_start_time = time.time()

        with torch.no_grad():
            for batch_num, batch in enumerate(val_dataloader):
                # MOVE DATA TO CUDA DEVICE
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = final_model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_collected.extend(labels.cpu().numpy())


        avg_val_loss = total_val_loss / len(val_dataloader)
        val_end_time = time.time()
        print(f"\nEpoch {epoch+1} Validation Summary:")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print("Validation Metrics:")
        val_accuracy, val_f1_weighted = calculate_metrics(val_labels_collected, val_preds, target_names)
        print(f"Validation Phase Time: {val_end_time - val_start_time:.2f} seconds")


        epoch_end_time = time.time()
        print("-" * 40)
        print(f"Epoch {epoch+1} TOTAL TIME: {epoch_end_time - epoch_start_time:.2f} seconds")
        print("-" * 40)

        # --- Early Stopping Check ---
        current_metric = avg_val_loss if EARLY_STOPPING_METRIC == 'val_loss' else val_f1_weighted
        improved = False
        if EARLY_STOPPING_METRIC == 'val_loss':
            if current_metric < best_val_metric:
                improved = True
        else: # Higher F1 is better
             if current_metric > best_val_metric:
                 improved = True

        if improved:
            print(f"*** Validation {EARLY_STOPPING_METRIC} improved ({best_val_metric:.4f} --> {current_metric:.4f}). Saving model... ***")
            best_val_metric = current_metric
            epochs_no_improve = 0
            best_model_state_dict = copy.deepcopy(final_model.state_dict())
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

    # 5. Final Evaluation
    if best_model_state_dict:
        print("\nLoading best model state for final evaluation...")
        final_model.load_state_dict(best_model_state_dict)
        final_model.to(device) # Ensure model is on CUDA after loading
    else:
        print("\nWarning: No best model state saved. Evaluating the last state.")
        final_model.to(device) # Ensure model is on CUDA

    print("\n--- Final Evaluation on Validation Set (Using Best Model) ---")
    final_model.eval()
    all_predictions = []
    all_labels = []
    eval_start_time = time.time()

    with torch.no_grad():
        for batch in val_dataloader:
            # MOVE DATA TO CUDA DEVICE
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = final_model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    eval_end_time = time.time()
    print("Final Classification Report:")
    final_accuracy, final_f1_weighted = calculate_metrics(all_labels, all_predictions, target_names)
    print(f"Final Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")

    # Optional: Save the best model
    SAVE_PATH = f"./{BERT_MODEL_NAME.replace('/', '_')}_best_model_undersampled_cuda_optuna.pth" # Modified path for optuna model
    if best_model_state_dict:
        torch.save(best_model_state_dict, SAVE_PATH)
        print(f"\nBest model saved to {SAVE_PATH}")