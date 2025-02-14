# tcn models for diabetes prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from pattention import SelfAttention

class RNNWithAttention(nn.Module):
    def __init__(self, 
        static_feature_dim,
        dynamic_feature_dim,
        sequence_length,
        num_layers,
        hidden_dim,
        num_classes,
        use_attention=True
    ):
        super(RNNWithAttention, self).__init__()

        # Self-Attention layer for dynamic data
        self.attention = SelfAttention(d_model=dynamic_feature_dim)
        self.use_attention = use_attention
        
        # RNN layer for dynamic data
        self.rnn = nn.LSTM(input_size=dynamic_feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected layer for static data
        self.fc_static = nn.Linear(static_feature_dim, hidden_dim)

        # Fully connected layer for combined data
        self.fc_combined = nn.Linear(hidden_dim*2, hidden_dim)

        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.5)


    def forward(self, static_data, dynamic_data):
        if self.use_attention:
            dynamic_data, _ = self.attention(dynamic_data)

        rnn_out, _ = self.rnn(dynamic_data)
        rnn_out = rnn_out[:, -1, :]
        
        static_out = F.relu(self.fc_static(static_data))
        
        combined = torch.cat((rnn_out, static_out), dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        
        return self.classifier(combined)


class GlycemicControlRNN(nn.Module):
    def __init__(self, 
        static_feature_dim, 
        dynamic_feature_dim, 
        sequence_length=36, 
        num_layers=3, 
        hidden_dim=64,  
        use_attention=True
    ):
        super(GlycemicControlRNN, self).__init__()
        self.rnn_with_attention = RNNWithAttention(
            static_feature_dim=static_feature_dim,
            dynamic_feature_dim=dynamic_feature_dim,
            sequence_length=sequence_length,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_classes=2,
            use_attention=use_attention
        )

    def forward(self, static_data, dynamic_data):
        return self.rnn_with_attention(static_data, dynamic_data)