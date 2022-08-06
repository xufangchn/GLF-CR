import os
import time
from metrics import *

class Generic_train_test():
	def __init__(self, model, opts, dataloader):
		self.model=model
		self.opts=opts
		self.dataloader=dataloader

	def decode_input(self, data):
		raise NotImplementedError()

	def train(self):
		total_steps = 0
		print('#training images ', len(self.dataloader)*self.opts.batch_sz)

		log_loss = 0

		for epoch in range(self.opts.max_epochs):
			print(f'epoch: {epoch}')

			for _, data in enumerate(self.dataloader):
				total_steps+=1
				_input=self.decode_input(data)

				self.model.set_input(_input)
				batch_loss = self.model.optimize_parameters()
				log_loss = log_loss + batch_loss

				#=========== visualize results ============#
				if total_steps % self.opts.log_freq==0:
					info = self.model.get_current_scalars()
					print('epoch', epoch, 'steps', total_steps, 'loss', log_loss/self.opts.log_freq, info)
					log_loss = 0

			if epoch % self.opts.save_freq==0:
				self.model.save_checkpoint(epoch)

			if epoch > self.opts.lr_start_epoch_decay - self.opts.lr_step:
				self.model.update_lr()