from .new_train import new_training
from .resume_train import resume_training




TRAINING_MODES = {
    'new': new_training,
    'resume': resume_training
}