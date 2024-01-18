from api.pipe_tasks.base import Base
from utils.datadir import project_dir, dress_worker_input_history, dress_worker_output_history


class InputWorkerData (Base):

    def __call__(self, *args, **kwargs):
        self.before()

    async def action(self):
        self.before()







