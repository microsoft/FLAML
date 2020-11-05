import json
import ast
from typing import IO
from contextlib import contextmanager


class TrainingLogRecord(object):

    def __init__(self,
                 record_id: int,
                 iter: int,
                 logged_metric: float,
                 train_time: float,
                 time_from_start: float,
                 config,
                 objective2minimize,
                 previous_best_val_loss,
                 previous_best_config,
                 move,
                 sample_size,
                 base,
                 config_sig):
        self.record_id = record_id
        self.iter = iter
        self.logged_metric = logged_metric
        self.train_time = train_time
        self.time_from_start = time_from_start
        self.config = config
        self.objective2minimize = objective2minimize
        self.previous_best_val_loss = previous_best_val_loss
        self.previous_best_config = previous_best_config
        self.move = move
        self.sample_size = sample_size
        self.base = base
        self.config_sig = config_sig

    def dump(self, fp: IO[str], encoding='utf-8'):
        d = vars(self)
        return json.dump(d, fp, encoding=encoding)

    @classmethod
    def load(cls, json_str):
        d = json.loads(json_str)
        return cls(**d)


class TrainingLogCheckPoint(TrainingLogRecord):

    def __init__(self, curr_best_record_id: int):
        self.curr_best_record_id = curr_best_record_id


class TrainingLogWriter(object):

    def __init__(self, automl_name: str, output_filename: str):
        self.automl_name = automl_name
        self.output_filename = output_filename
        self.file = None
        self.current_best_loss_info = None
        self.current_best_loss = float('+inf')
        self.current_sample_size = None
        self.current_record_id = 0

    def open(self):
        self.file = open(self.output_filename, 'w')

    def append(self,
               it_counter: int,
               train_loss: float,
               train_time: float,
               all_time: float,
               i_config,
               val_loss,
               best_val_loss,
               best_config,
               current_config_arr_iter,
               move,
               sample_size,
               base='None',
               config_sig='None'):
        if self.file is None:
            raise IOError("Call open() to open the outpute file first.")
        if val_loss is None:
            raise ValueError('TEST LOSS NONE ERROR!!!')
        record = TrainingLogRecord(self.current_record_id,
                                   it_counter,
                                   train_loss,
                                   train_time,
                                   all_time,
                                   i_config,
                                   val_loss,
                                   best_val_loss,
                                   best_config,
                                   move,
                                   sample_size,
                                   base,
                                   config_sig)
        if val_loss < self.current_best_loss or \
            val_loss == self.current_best_loss and \
                sample_size > self.current_sample_size:
            self.current_best_loss = val_loss
            self.current_sample_size = sample_size
            self.current_best_loss_info = self.current_record_id
        self.current_record_id += 1
        record.dump(self.file)
        self.file.write('\n')
        self.file.flush()

    def checkpoint(self):
        if self.file is None:
            raise IOError("Call open() to open the outpute file first.")
        if not self.current_best_loss_info:
            raise Exception("checkpoint() called before "
                            "any record is written.")
        record = TrainingLogCheckPoint(self.current_best_loss_info)
        record.dump(self.file)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


class TrainingLogReader(object):

    def __init__(self, filename: str):
        self.filename = filename
        self.file = None

    def open(self):
        self.file = open(self.filename)

    def records(self):
        if self.file is None:
            raise IOError("Call open() before reading log file.")
        for line in self.file:
            data = json.loads(line)
            if len(data) == 1:
                # Skip checkpoints.
                continue
            yield TrainingLogRecord(**data)

    def close(self):
        self.file.close()


@contextmanager
def training_log_reader(filename: str):
    try:
        r = TrainingLogReader(filename)
        r.open()
        yield r
    finally:
        r.close()
