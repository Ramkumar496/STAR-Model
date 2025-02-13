import os
import pandas as pd
import numpy as np
from math import ceil
import random
from copy import deepcopy

class Dataset(object):
    def __init__(self, data=None, data_path="", args=None, n_items=0, train_set=True):
        args = args or {}
        self.batch_size = getattr(args, "batch_size", 32)
        self.train_set = train_set
        self.data_name = getattr(args, "data_name", "dataset")
        self.n_items = n_items
        self.padd_idx = n_items
        self.set_keys()

        # Load or assign data
        self.data = data if data is not None else self.load_data(data_path)

        # Sort and prepare data
        self.sort_data()
        self.max_session_len = self.get_max_sequence_len(self.data)

    def load_data(self, data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_pickle(data_path)
        print(f"Data loaded. Shape: {data.shape}")
        return data

    def set_keys(self):
        self.session_key = "session_id"
        self.item_idx = "item_idx"
        self.time_key = "time"

    def sort_data(self):
        self.data.sort_values([self.session_key, self.time_key], inplace=True)
        self.data = self.data.reset_index(drop=True)
        print(f"Data sorted by {self.session_key} and {self.time_key}.")

    def create_offset_sessions(self, data):
        offset_sessions = (data[self.session_key].values
                          - np.roll(data[self.session_key].values, 1))
        offset_sessions = np.nonzero(offset_sessions)[0]
        print(f"Offset sessions created: {len(offset_sessions)} sessions.")
        return offset_sessions

    def get_max_sequence_len(self, data):
        max_len = max(data.groupby(self.session_key).size())
        print(f"Max session length: {max_len}")
        return max_len

    def sort_session_by_len(self, data):
        size_key = "session_size"
        session_len = data.groupby(self.session_key).size().sort_values()
        session_len = session_len.to_frame(name=size_key)
        data = pd.merge(data, session_len, on=self.session_key, how="inner")
        data.sort_values([size_key, self.session_key, self.time_key], inplace=True)
        data = data.reset_index(drop=True)
        return data

    def get_len_offsets(self, data):
        data = self.sort_session_by_len(data)
        size_key = "session_size"
        lengths = data[size_key].values
        lengths = np.roll(lengths, 1)
        lengths = lengths - data[size_key].values
        len_change_idx = np.sort(np.nonzero(lengths != 0)[0])
        offset_lengths = data[size_key].values[len_change_idx]
        del data[size_key]
        print(f"Offset lengths: {len(offset_lengths)}")
        return data, np.vstack((offset_lengths, len_change_idx)).T

    def __iter__(self):
        """Returns the iterator for producing mini-batches."""
        yield from self.get_mini_batch()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset."""
        row = self.data.iloc[idx]
        
        # Extract item sequence and target
        item_sequence = row[self.item_idx]
        target = row[self.item_idx]  # You may need to update this depending on the target logic.

        # Print the shapes and content of item_sequence and target
        #print(f"Sample {idx}: item_sequence={item_sequence}, target={target}")
        #print(f"Shape of item_sequence: {np.shape(item_sequence)}, Shape of target: {np.shape(target)}")
        
        # Extract other components
        h_a, m_a, s_a = row['h_a'], row['m_a'], row['s_a']
        h_b, m_b, s_b = row['h_b'], row['m_b'], row['s_b']

        # Debugging: Print the shape and content of each component
        #print(f"h_a: {np.shape(h_a)}, m_a: {np.shape(m_a)}, s_a: {np.shape(s_a)}")
        #print(f"h_b: {np.shape(h_b)}, m_b: {np.shape(m_b)}, s_b: {np.shape(s_b)}")

        # Return all 8 components
        return item_sequence, target, h_a, m_a, s_a, h_b, m_b, s_b

    def get_mini_batch(self):
        """Generates mini-batches of data."""
        lengths_list = np.random.permutation(np.arange(2, self.max_session_len + 1))
        data, start_indices = self.get_len_offsets(self.data)
        offset_sessions = self.create_offset_sessions(data)
        max_session_len = self.get_max_sequence_len(data)

        item_idx = data[self.item_idx].values
        h_a_idx = data['h_a'].values
        m_a_idx = data['m_a'].values
        s_a_idx = data['s_a'].values
        h_b_idx = data['h_b'].values
        m_b_idx = data['m_b'].values
        s_b_idx = data['s_b'].values

        for length in lengths_list:
            if length > max_session_len:
                continue

            start_index = np.argmax(start_indices[:, 0] >= length)
            start_index = start_indices[start_index, 1]
            session_starts = offset_sessions[offset_sessions >= start_index]
            np.random.shuffle(session_starts)
            batch_size = min(self.batch_size, len(session_starts))

            while len(session_starts) >= batch_size:
                batch_sessions = session_starts[:batch_size]
                session_starts = session_starts[batch_size:]

                # Ensure batch_sessions contains start and end tuples (start, end)
                batch_sessions = [(start, start + length) for start in batch_sessions]

                #print(f"Batch created. Batch sessions: {batch_sessions}")

                # Now unpack the sessions properly
                item_sequence = np.array([item_idx[start:end] for start, end in batch_sessions])
                # In DataSet.py, within the get_mini_batch function
                targets = np.array([item_idx[min(end, len(item_idx) - 1)] for start, end in batch_sessions])


                # Ensure the batch has 8 components per sample
                #print(f"Batch item_sequence shape: {item_sequence.shape}, targets shape: {targets.shape}")
                assert len(item_sequence.shape) == 2, "Item sequence should be a 2D array."
                assert len(targets.shape) == 1, "Targets should be a 1D array."

                yield item_sequence, targets

    def get_validation(self, partition):
        """Splits and returns validation data."""
        if self.train_set:
            n_sequence = int((len(self.data) - self.data[self.session_key].nunique()) / (partition * 100))
            session_ids = self.data[self.session_key].values
            
            # Fixed generator expression with proper else condition
            split_index = next(
                (i for i, (cur_id, prev_id) in enumerate(zip(session_ids[::-1], session_ids[::-1][1:]))
                if cur_id != prev_id and i >= n_sequence),
                len(self.data)  # Default value if no match is found
            )

            valid_data = deepcopy(self.data.iloc[split_index:])
            self.data = deepcopy(self.data.iloc[:split_index])
            print(f"Validation data split: {len(valid_data)} samples.")
            return valid_data
        else:
            raise ValueError("Validation not applicable for test datasets.")


    def sort_sessions(self, data):
        """Sort sessions by their start time."""
        session_times = data.groupby(self.session_key)[self.time_key].min()
        session_times = pd.DataFrame({self.session_key: session_times.index,
                                      'session_time': session_times.values})
        data = pd.merge(data, session_times, on=self.session_key, how="inner")
        data.sort_values(['session_time', self.session_key, self.time_key], inplace=True)
        data = data.reset_index(drop=True)
        del data['session_time']
        return data


def create_sets(data_dir, args, n_items):
    """Creates train and test dataset objects."""
    train_path = os.path.join(data_dir, f"train_1_{args.fraction}.pkl") if args.data_name == "yoochoose" \
        else os.path.join(data_dir, "train_1_1.pkl")

    train_dataset = Dataset(data_path=train_path, args=args, n_items=n_items)

    if not args.validation:
        test_path = os.path.join(data_dir, "test.pkl")
        test_dataset = Dataset(data_path=test_path, args=args, n_items=n_items, train_set=False)
    else:
        valid_data = train_dataset.get_validation(args.valid_portion)
        test_dataset = Dataset(data=valid_data, args=args, n_items=n_items, train_set=False)

    return train_dataset, test_dataset
