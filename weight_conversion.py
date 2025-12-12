import torch
state_dict = torch.load('assets/reward_model.pt')

new_state_dict = {}

for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key.replace('model.', 'model.language_model.')
    elif key.startswith('visual.'):
        new_key = key.replace('visual.', 'model.visual.')
    else:
        raise ValueError(f'Unexpected key {key}')
    new_state_dict[new_key] = value

torch.save(new_state_dict, 'assets/reward_model.pt')