import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    def __init__(self, upstream_model='wav2vec2', hidden_state=12, num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model) # loading ssl model from s3prl
        self.n_encoder_layer = len(self.upstream.model.encoder.layers)
        assert hidden_state > 0 and hidden_state <= self.n_encoder_layer 
        self.hidden_state = 'hidden_state_{}'.format(hidden_state)
        
        for param in self.upstream.parameters():
            param.requires_grad = True
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        
        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x_input = [torch.narrow(wav, 0, 0, x_len[i]) for (i, wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x_input)
        # s3prl sometimes misses the ouput of some hidden states so need a while loop to assure we can get the output of the desired hidden state 
        while self.hidden_state not in x.keys():
            x = self.upstream(x_input)
        x = x[self.hidden_state]
        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        xM = self.dropout(torch.cat((torch.mean(xM, dim=1), torch.std(xM, dim=1)), dim=1))
        xF = self.dropout(torch.cat((torch.mean(xF, dim=1), torch.std(xF, dim=1)), dim=1))
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        age = self.age_regressor(output)
        return age, gender
