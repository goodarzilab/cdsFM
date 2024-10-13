import torch
from transformers import StoppingCriteria


class SEPStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, verbose=False):
        super().__init__()
        self.eos_token = tokenizer.sep_token_id
        self.done = None
        self.eos_index = None
        self.verbose = verbose

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        batch_size, seq_len = input_ids.shape

        if self.done == None:
            self.done = torch.zeros(
                batch_size, dtype=torch.bool, device=input_ids.device
            )
            self.eos_index = torch.zeros(
                batch_size, dtype=torch.int, device=input_ids.device
            )

        last_ids = input_ids[:, -1]

        # Create mask of where the last token is EOS
        done_update = self.done | (last_ids == self.eos_token)

        # Store the indices where we stopped at for each sequence in the batch.
        # Where the 'done' state has changed, store the seq_len (last index), else 0
        eos_index_update = torch.where(
            done_update ^ self.done, torch.full_like(self.eos_index, seq_len), 0
        )

        # Add the update to the indices
        self.eos_index += eos_index_update

        # Update the done flags
        self.done = done_update

        if self.verbose:
            if len(scores) % 200 == 0:
                print(f"EOS iter {len(scores)}: n_stopped: {self.done.sum()}")

        return self.done.all()
