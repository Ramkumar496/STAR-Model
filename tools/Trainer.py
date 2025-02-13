import torch
import torch.optim as optim
import time
import logging
import numpy as np
from torch.utils.data import DataLoader
from tools.Metrics import calc  # Import the calc function for metrics calculation

class Trainer:
    def __init__(self, device='cpu'):  # Default to CPU if no device is specified
        self.device = torch.device(device)

    def tensorize_batch(self, batch):
        if len(batch) == 2:
            item_sequence, target = batch
            item_sequence = torch.tensor(item_sequence).to(self.device)
            target = torch.tensor(target).to(self.device).long().unsqueeze(0)
            h_a_o = torch.zeros_like(item_sequence).to(self.device)
            m_a_o = torch.zeros_like(item_sequence).to(self.device)
            s_a_o = torch.zeros_like(item_sequence).to(self.device)
            h_b_o = torch.zeros_like(item_sequence).to(self.device)
            m_b_o = torch.zeros_like(item_sequence).to(self.device)
            s_b_o = torch.zeros_like(item_sequence).to(self.device)
        else:
            item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o = batch
            item_sequence = torch.tensor(item_sequence).to(self.device)
            target = torch.tensor(target).to(self.device).long().unsqueeze(0)
            h_a_o = torch.tensor(h_a_o).to(self.device)
            m_a_o = torch.tensor(m_a_o).to(self.device)
            s_a_o = torch.tensor(s_a_o).to(self.device)
            h_b_o = torch.tensor(h_b_o).to(self.device)
            m_b_o = torch.tensor(m_b_o).to(self.device)
            s_b_o = torch.tensor(s_b_o).to(self.device)

        return item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o

    def train(self, model, train_data, valid_data, args, result_dir):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

        best_valid_loss = float('inf')
        logging.basicConfig(filename=f'{result_dir}/training_log.txt', level=logging.INFO)

        for epoch in range(args.epoch):
            start_time = time.time()
            model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_data):
                item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o = self.tensorize_batch(batch)
                optimizer.zero_grad()
                loss = model.fit(item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                logging.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_data)}, Loss: {loss.item()}")
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_data)}, Loss: {loss.item()}")

                if args.max_batches is not None and batch_idx >= args.max_batches:
                    print(f"Debugging: Stopped after {args.max_batches} batches.")
                    break

            scheduler.step()
            epoch_time = time.time() - start_time

            valid_loss, recall, mrr, accuracy, precision = self.eval(model, valid_data, args)
            print(f'Epoch {epoch+1} '
                  f'Recall: {recall:.4f}, MRR: {mrr:.4f}, '
                  f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Time: {epoch_time:.2f}s')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'{result_dir}/best_model.pth')

    def eval(self, model, valid_data, args):
        model.eval()
        total_loss = 0
        recall_list = []
        mrr_list = []
        accuracy_list = []
        precision_list = []

        with torch.no_grad():
            for batch in valid_data:
                item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o = self.tensorize_batch(batch)
                output = model.forward(item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)

                # Compute metrics using logits and targets
                metrics = calc(output, target, k=args.top_k)

                total_loss += metrics.get('loss', 0.0)  # Assuming `calc` also returns a loss value if needed
                recall_list.append(metrics['Recall@K'])
                mrr_list.append(metrics['MRR@K'])
                accuracy_list.append(metrics['Accuracy'])
                precision_list.append(metrics['Precision@K'])

        average_loss = total_loss / len(valid_data)
        average_recall = np.mean(recall_list)
        average_mrr = np.mean(mrr_list)
        average_accuracy = np.mean(accuracy_list)
        average_precision = np.mean(precision_list)

        return average_loss, average_recall, average_mrr, average_accuracy, average_precision

class NewTrainer:
    def __init__(self, dataset, model, optimizer, args, device='cpu'):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = torch.device(device)
        
    def tensorize_batch(self, batch):
        # This method assumes that the batch has 2 elements (item_sequence and target)
        if len(batch) == 2:
            item_sequence, target = batch
            item_sequence = torch.tensor(item_sequence).to(self.device)
            target = torch.tensor(target).to(self.device).long().unsqueeze(0)  # Add dimension if necessary
            # Create placeholders for other elements as necessary (fill them with zeros or other values)
            h_a_o = torch.zeros_like(item_sequence).to(self.device)
            m_a_o = torch.zeros_like(item_sequence).to(self.device)
            s_a_o = torch.zeros_like(item_sequence).to(self.device)
            h_b_o = torch.zeros_like(item_sequence).to(self.device)
            m_b_o = torch.zeros_like(item_sequence).to(self.device)
            s_b_o = torch.zeros_like(item_sequence).to(self.device)
        else:
            # This block handles the case when the batch has 8 elements
            item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o = batch
            item_sequence = torch.tensor(item_sequence).to(self.device)
            target = torch.tensor(target).to(self.device).long().unsqueeze(0)
            h_a_o = torch.tensor(h_a_o).to(self.device)
            m_a_o = torch.tensor(m_a_o).to(self.device)
            s_a_o = torch.tensor(s_a_o).to(self.device)
            h_b_o = torch.tensor(h_b_o).to(self.device)
            m_b_o = torch.tensor(m_b_o).to(self.device)
            s_b_o = torch.tensor(s_b_o).to(self.device)

        return item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o

    def train(self):
        # Assuming the dataset is a DataLoader or similar iterable
        train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        best_valid_loss = float('inf')

        for epoch in range(self.args.epoch):
            self.model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Debugging: Print the structure of the batch before tensorizing
                #print(f"Batch {batch_idx} structure:", len(batch), batch)

                # Process the batch
                item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o = self.tensorize_batch(batch)

                # Zero gradients before backpropagation
                self.optimizer.zero_grad()

                # Get the model's loss and metrics
                loss, recall, mrr = self.model.fit(item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o, self.args.batch_size)
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Log progress
                if (batch_idx + 1) % self.args.log_interval == 0:
                    print(f"Epoch [{epoch+1}/{self.args.epoch}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # After each epoch, evaluate the model
            valid_loss, recall, mrr = self.eval(self.model, self.args)

            print(f"Epoch [{epoch+1}/{self.args.epoch}], Train Loss: {total_loss / len(train_loader):.4f}, "
                  f"Validation Loss: {valid_loss:.4f}, Recall: {recall:.4f}, MRR: {mrr:.4f}")

            # Save model if it improves validation loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

    def eval(self, model, args):
        model.eval()
        total_loss = 0
        recall_list = []
        mrr_list = []

        with torch.no_grad():
            for batch in self.dataset:
                item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o = self.tensorize_batch(batch)

                # Get the model's output
                output = model(item_sequence, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o)
                
                # Calculate loss and metrics
                loss, recall, mrr = model.test(item_sequence, target, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o, args.batch_size, args.top_k)
                total_loss += loss.item()
                recall_list.append(recall)
                mrr_list.append(mrr)

        average_loss = total_loss / len(self.dataset)
        average_recall = np.mean(recall_list)
        average_mrr = np.mean(mrr_list)
        return average_loss, average_recall, average_mrr
