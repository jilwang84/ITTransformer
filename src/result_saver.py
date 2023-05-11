# Copyright (c) 2023-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from .base.result import result
import torch


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_file_path = None

    # Save learned model
    def save_learned_model(self):
        print('Saving model...')
        torch.save(self.data, self.result_destination_file_path + '.pth')
        print('Saving model done!')


# Save model's training loss
def save_training_loss(training_loss, path):
    print('Saving training loss...')
    with open(path, 'w') as fp:
        for epoch, loss in enumerate(training_loss):
            fp.write("%s,%s\n" % (str(epoch + 1), str(loss)))
        print('Saving training loss done!')

