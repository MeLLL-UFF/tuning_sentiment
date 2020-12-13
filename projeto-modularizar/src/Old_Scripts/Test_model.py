from fairseq.models.roberta import RobertaModel
import torch
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint3.pt', 'data-bin/dataset')
print(isinstance(roberta.model, torch.nn.Module))
