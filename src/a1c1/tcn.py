# tcn models for diabetes prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from pattention import SelfAttention

class TCNWithAttention(nn.Module):
    def __init__(self, 
        static_feature_dim, # static feature dimension 
        dynamic_feature_dim, # dynamic feature dimension
        sequence_length,  # sequence length
        tcn_channels,  # list of integers with the number of channels in each layer
        num_classes, # number of classes for the classification task
        static_hidden_dim=12, # hidden layer dimension for static data
        combined_hidden_dim=48, # hidden layer dimension for combined data
        dropout_rate=0.5, # dropout rate
        kernel_size=3, # kernel size for TCN
        padding=1, # padding for TCN
        use_attention=True # use attention layer
    ):
        super(TCNWithAttention, self).__init__()
        self.dropout_rate = dropout_rate

        # Self-Attention layer for dynamic data
        self.attention = SelfAttention(d_model=dynamic_feature_dim)
        self.use_attention = use_attention

        # TCN layers for dynamic data
        self.tcn_layers = nn.ModuleList()
        in_channels = dynamic_feature_dim
        for out_channels in tcn_channels:
            self.tcn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    padding=padding
                )
            )
            in_channels = out_channels  # Update in_channels for the next layer

        # Fully connected layer for static data
        self.fc_static = nn.Linear(static_feature_dim, static_hidden_dim)

        # Dropout after the static layer
        self.dropout_static = nn.Dropout(dropout_rate)

        # Fully connected layer for combined data
        self.fc_combined = nn.Linear(tcn_channels[-1] * sequence_length + static_hidden_dim, combined_hidden_dim)

        # Dropout after the combined layer
        self.dropout_combined = nn.Dropout(dropout_rate)

        # Final classification layer
        self.classifier = nn.Linear(combined_hidden_dim, num_classes)

    def forward(self, static_data, dynamic_data):
        # static_data: (batch_size, static_feature_dim)
        # dynamic_data: (batch_size, sequence_length, dynamic_feature_dim)

        # Apply self-attention to dynamic data
        if self.use_attention:
            attention_output, _ = self.attention(dynamic_data)
            # Apply TCN layers to dynamic data
            x = attention_output.permute(0, 2, 1)  # order change for Conv1d (batch_size, dynamic_feature_dim, sequence_length)
        else:
            x = dynamic_data.permute(0, 2, 1)

        for conv_layer in self.tcn_layers:
            x = F.relu(conv_layer(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Dropout after each layer

        # Flatten the output of the last TCN layer
        x = x.view(x.size(0), -1)  # (batch_size, tcn_channels[-1] * sequence_length)

        # Apply fully connected layer to static data
        static_features = F.relu(self.fc_static(static_data))
        static_features = self.dropout_static(static_features)  # Dropout after the static layer

        # Combine the outputs of TCN and static data
        combined = torch.cat([x, static_features], dim=1)  # Concatenate along the feature dimension

        # Apply fully connected layer to the combined data
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout_combined(combined)  # Dropout after the combined layer

        # Apply final classification layer
        output = self.classifier(combined)
        return output

class GlycemicControl(nn.Module):
    def __init__(self,
            static_feature_dim=24, # static feature dimension 
            dynamic_feature_dim=256, # dynamic feature dimension, labs, medications, vitals
            sequence_length=36,  # sequence length 3 years by month
            tcn_channels=[128, 64, 32],  # list of integers with the number of channels in each layer
            kernel_size=3, # kernel size for TCN
            padding=1, # padding for TCN
            combined_hidden_dim=48, # hidden layer dimension for combined data
            static_hidden_dim=12, # hidden layer dimension for static data
            use_attention=True # use attention layer
        ):
        super(GlycemicControl, self).__init__()
        self.tcn_with_attention = TCNWithAttention(
            static_feature_dim=static_feature_dim, # static feature dimension 
            dynamic_feature_dim=dynamic_feature_dim, # dynamic feature dimension, labs 
            sequence_length=sequence_length,  # sequence length
            tcn_channels=tcn_channels,  # list of integers with the number of channels in each layer
            dropout_rate=0.5, # dropout rate
            num_classes=2, # number of classes for the classification task
            kernel_size=kernel_size, # kernel size for TCN
            padding=padding, # padding for TCN
            combined_hidden_dim=combined_hidden_dim, # hidden layer dimension for combined data
            static_hidden_dim=static_hidden_dim, # hidden layer dimension for static data
            use_attention=use_attention # use attention layer
        )

    def forward(self, stat_data, seq_data):
        # Self-Attention and TCN processing on sequential data
        out = self.tcn_with_attention(stat_data, seq_data)
        return out
