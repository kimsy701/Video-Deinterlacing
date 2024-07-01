import pandas as pd


# with open('/home/kimsy701/deinter/load_models/deinter_large/trained-models/model-step2-i1.pkl', 'rb') as f:
#     data1 = pickle.load(f)
    
# with open('/home/kimsy701/deinter/load_models/deinter_large/trained-models/model-step2-i2241-psnr22.24337387084961.pkl', 'rb') as f:
#     data2 = pickle.load(f)
    
unpickled_df1 = pd.read_pickle("/home/kimsy701/deinter/load_models/deinter_large/trained-models/model-step2-i1.pkl")  
print(unpickled_df1[0][0][0])
unpickled_df2 = pd.read_pickle("/home/kimsy701/deinter/load_models/deinter_large/trained-models/model-step2-i2241-psnr22.24337387084961.pkl")  
print(unpickled_df2[0][0][0])




def load_pretrained_state_dict(model, model_file):
    if (model_file == "") or (not os.path.exists(model_file)):
        raise ValueError(
                "Please set the correct path for pretrained model!")

    print("Load pretrained model from %s."  % model_file)
    rand_state_dict = model.state_dict()
    pretrained_state_dict = torch.load(model_file)

    return Pipeline.convert_state_dict(
            rand_state_dict, pretrained_state_dict)

state_dict = load_pretrained_state_dict(self.model, model_file)
self.model.load_state_dict(state_dict)
