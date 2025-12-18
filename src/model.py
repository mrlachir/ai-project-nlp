import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [B, S]
        embedded = self.dropout(self.embedding(src))  # [B, S, E]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [B, S, H]
        return outputs, hidden, cell


class Attention(nn.Module):
    """
    Additive attention (Bahdanau-style).
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.W_h = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_s = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [B, S, H]
        # decoder_hidden:  [B, H]   (top layer hidden)
        B, S, H = encoder_outputs.shape

        dec = decoder_hidden.unsqueeze(1).expand(B, S, H)  # [B, S, H]
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(dec))  # [B, S, H]
        scores = self.v(energy).squeeze(-1)  # [B, S]
        attn_weights = F.softmax(scores, dim=1)  # [B, S]
        return attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx, attention: Attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = attention
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: [B]  (one step)
        # hidden: [L, B, H]
        # cell:   [L, B, H]
        # encoder_outputs: [B, S, H]

        input_token = input_token.unsqueeze(1)  # [B, 1]
        embedded = self.dropout(self.embedding(input_token))  # [B, 1, E]

        dec_hidden_top = hidden[-1]  # [B, H]
        attn_weights = self.attention(encoder_outputs, dec_hidden_top)  # [B, S]
        attn_weights = attn_weights.unsqueeze(1)  # [B, 1, S]

        context = torch.bmm(attn_weights, encoder_outputs)  # [B, 1, H]

        rnn_input = torch.cat((embedded, context), dim=2)  # [B, 1, E+H]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [B, 1, H]

        output = output.squeeze(1)   # [B, H]
        context = context.squeeze(1) # [B, H]
        embedded = embedded.squeeze(1) # [B, E]

        pred = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [B, V]
        return pred, hidden, cell, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: [B, S]
        tgt: [B, T] (includes <SOS> ... <EOS>)
        """
        B, T = tgt.shape
        V = self.decoder.fc_out.out_features

        outputs = torch.zeros(B, T, V, device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input_token = tgt[:, 0]  # first token is <SOS>

        for t in range(1, T):
            pred, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = pred

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    @torch.no_grad()
    def greedy_translate(self, src, max_len=40):
        """
        src: [1, S]
        returns: list of token ids (including <SOS> and <EOS>)
        """
        self.eval()
        encoder_outputs, hidden, cell = self.encoder(src)

        input_token = torch.tensor([self.sos_idx], device=self.device)
        result = [self.sos_idx]

        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            next_token = pred.argmax(1).item()
            result.append(next_token)
            input_token = torch.tensor([next_token], device=self.device)
            if next_token == self.eos_idx:
                break

        return result
