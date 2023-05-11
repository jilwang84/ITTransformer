# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import abc


class result:
    """
    ResultModule: Abstract Class
    Entries: 
    """
    
    data = None
    
    result_name = None
    result_description = None
    
    result_destination_file_path = None
    
    # initialization function
    def __init__(self, rName=None, rType=None):
        self.result_name = rName
        self.result_description = rType

    @abc.abstractmethod
    def save(self):
        return
 
    @abc.abstractmethod
    def load(self):
        return

