import json
from typing import IO
from contextlib import contextmanager
import warnings

class ResultLogRecord(object):

    def __init__(self,
                 record_id: int,
                 y_predict: float,
                 y: float,
                 loss: float,
                 time_used: float,
                 incumbent_config: str=None,
                 champion_config: str=None,
                 ):
        self.record_id = record_id
        self.y_predict= float('{:.5f}'.format(float(y_predict)))
        self.y = float('{:.5f}'.format(float(y)))
        self.loss = float('{:.5f}'.format(float(loss)))
        self.time_used = float('{:.4f}'.format(float(time_used)))
        # self.inc = incumbent_config
        # self.cha = champion_config

    def dump(self, fp: IO[str]):
        d = vars(self)
        return json.dump(d, fp)

    @classmethod
    def load(cls, json_str: str):
        d = json.loads(json_str)
        return cls(**d)

    def __str__(self):
        return json.dumps(vars(self))

class ResultLogWriter(object):

    def __init__(self, output_filename: str, 
        loss_metric:str=None, 
        dataset_name:str=None,
        method_name:str=None,
        method_config:dict=None,):
        self.output_filename = output_filename
        self.file = None
        self.loss_metric=loss_metric
        self.dataset_name=dataset_name
        self.method_name=method_name
        self.method_config=method_config

    def open(self):
        self.file = open(self.output_filename, 'w')

    def append(self,
               record_id: int,
               y_predict: float,
               y: float,
               loss: float = None,
               time_used: float = None, 
               incumbent_config: str = None,
               champion_config: str = None):
        
        if self.file is None:
            raise IOError("Call open() to open the outpute file first.")
        record = ResultLogRecord(record_id,
                                y_predict,
                                y,
                                loss,
                                time_used,
                                incumbent_config,
                                champion_config)
        record.dump(self.file)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


class ResultLogReader(object):

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
            yield ResultLogRecord(**data)

    def close(self):
        self.file.close()

    def get_record(self, record_id) -> ResultLogRecord:
        if self.file is None:
            raise IOError("Call open() before reading log file.")
        for rec in self.records():
            if rec.record_id == record_id:
                return rec
        raise ValueError(f"Cannot find record with id {record_id}.")


@contextmanager
def training_log_writer(filename: str):
    try:
        w = ResultLogWriter(filename)
        w.open()
        yield w
    finally:
        w.close()


@contextmanager
def training_log_reader(filename: str):
    try:
        r = ResultLogReader(filename)
        r.open()
        yield r
    finally:
        r.close()
