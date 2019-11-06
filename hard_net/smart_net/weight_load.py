import torch

def load_weight(model,model_weights_path,brain_weights_path,brain = None):
	if model:
		try:
			model.load_state_dict(torch.load(model_weights_path), strict = True)
			print("weights are strictly used for model")
		except:
			try:
				model_pretrained_dict = torch.load(model_weights_path)
				for param_tensor in model.state_dict():
					model.state_dict()[param_tensor][:] = model_pretrained_dict["state_dict"][param_tensor][:]
				print("weights are not strictly used for model")
			except: pass
			print("randomly weights are choosen for model")

	if brain :
		try:
			brain.load_state_dict(torch.load(brain_weights_path), strict = True)
			print("weights are strictly used for brain")
		except:
			try:
				bain_rpretrained_dict = torch.load(brain_weights_path)
				for param_tensor in brain.state_dict():
					brain.state_dict()[param_tensor][:] = brain_pretrained_dict["state_dict"][param_tensor][:]
				print("weights are not strictly used for brain")
			except: pass
			print("randomly weights are choosen for brain")
