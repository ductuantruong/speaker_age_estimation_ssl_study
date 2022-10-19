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
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x_input = [torch.narrow(wav, 0, 0, x_len[i]) for (i, wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x_input)
        # s3prl sometimes misses the ouput of some hidden states so need a while loop to assure we can get the output of the desired hidden state 
        while self.hidden_state not in x.keys():
            x = self.upstream(x_input)
        x = x[self.hidden_state]
        x = self.transformer_encoder(x)
        x = self.dropout(torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1))
        x = self.dropout(self.fc(x))
        gender = self.gender_classifier(x)
        age = self.age_regressor(x)
        return age, gender
