import torch
import torch.nn as nn
from transformers import BertModel

class MultimodalFusion(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalFusion, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.cnn1d = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3)

        # Update LSTM input size to match CNN output
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=79, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.fc_text = nn.Linear(256 * (128 // 3), 128)
        self.fc_voice = nn.Linear(2 * hidden_size, 128)  # Corrected dimension

        self.fc_final = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask, voice_input):
        # Text Path
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = self.pool(torch.relu(self.cnn1d(bert_out.permute(0, 2, 1)))).view(input_ids.size(0), -1)
        text_feat = self.fc_text(text_feat)

        # Voice Path
        lstm_out, _ = self.lstm(voice_input)  # (batch_size, seq_len, 2 * hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the last timestep (batch_size, 2 * hidden_size)
        voice_feat = self.fc_voice(lstm_out)

        # Fusion
        fused = torch.cat((text_feat, voice_feat), dim=1)
        output = self.fc_final(fused)

        return output
