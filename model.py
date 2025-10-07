import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Combine forward and backward directions
        hidden = torch.cat([hidden[::2], hidden[1::2]], dim=2)  # [n_layers, batch, hidden_dim*2]
        cell   = torch.cat([cell[::2],   cell[1::2]],   dim=2)
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, enc_layers=2, n_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Decoder expects hidden_dim*2 from Encoder, but can have more layers
        self.lstm = nn.LSTM(emb_dim, hidden_dim * 2, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.enc_layers = enc_layers
        self.dec_layers = n_layers

    def forward(self, x, hidden, cell):
        # Expand encoder's hidden/cell to match decoder layer count if needed
        if hidden.size(0) < self.dec_layers:
            diff = self.dec_layers - hidden.size(0)
            hidden = torch.cat([hidden, hidden[-1:].repeat(diff, 1, 1)], dim=0)
            cell = torch.cat([cell, cell[-1:].repeat(diff, 1, 1)], dim=0)

        x = x.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)

        enc_out, hidden, cell = self.encoder(src, src_lens)
        input = tgt[:, 0]  # <SOS>

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

class TransliterationModel:
    def __init__(self, model_path, ur_vocab_path, en_vocab_path, device='cpu'):
        self.device = torch.device(device)
        self.ur_stoi, self.ur_itos, self.ur_vocab_size = self.load_vocab(ur_vocab_path)
        self.en_stoi, self.en_itos, self.en_vocab_size = self.load_vocab(en_vocab_path)
        
        # Initialize model architecture - MUST MATCH YOUR TRAINING CONFIG
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        
        enc = Encoder(self.ur_vocab_size, ENC_EMB_DIM, HID_DIM, n_layers=2, dropout=0.3)
        dec = Decoder(self.en_vocab_size, DEC_EMB_DIM, HID_DIM, enc_layers=2, n_layers=4, dropout=0.3)
        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"✅ Urdu vocab size: {self.ur_vocab_size}")
            print(f"✅ English vocab size: {self.en_vocab_size}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_vocab(self, path):
        """Load vocabulary from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            stoi = {ch: i for i, ch in enumerate(lines)}
            itos = {i: ch for i, ch in enumerate(lines)}
            vocab_size = len(lines)
            print(f"✅ Loaded vocabulary from {path} with {vocab_size} tokens")
            return stoi, itos, vocab_size
        except Exception as e:
            print(f"❌ Error loading vocabulary from {path}: {e}")
            raise
    
    def transliterate(self, sentence, max_len=50):
        """Transliterate Urdu text to Roman Urdu"""
        try:
            # Convert sentence to sequence of indices
            src_seq = [self.ur_stoi.get(ch, self.ur_stoi["<UNK>"]) for ch in sentence][:max_len]
            src_tensor = torch.tensor([src_seq], dtype=torch.long).to(self.device)
            src_lens = torch.tensor([len(src_seq)])
            
            with torch.no_grad():
                # Encode the input
                enc_out, hidden, cell = self.model.encoder(src_tensor, src_lens)
                
                # Start with <SOS> token
                input_token = torch.tensor([self.en_stoi["<SOS>"]], dtype=torch.long).to(self.device)
                result = []
                
                # Generate output sequence
                for _ in range(max_len):
                    output, hidden, cell = self.model.decoder(input_token, hidden, cell)
                    top1 = output.argmax(1).item()
                    
                    # Stop if <EOS> token is generated
                    if top1 == self.en_stoi["<EOS>"]:
                        break
                    
                    # Add character to result
                    result.append(self.en_itos[top1])
                    input_token = torch.tensor([top1], dtype=torch.long).to(self.device)
            
            return "".join(result)
            
        except Exception as e:
            print(f"❌ Error during transliteration: {e}")
            return f"Error: {str(e)}"

    def get_vocab_info(self):
        """Get information about loaded vocabularies"""
        return {
            "urdu_vocab_size": self.ur_vocab_size,
            "english_vocab_size": self.en_vocab_size,
            "urdu_special_tokens": {k: v for k, v in self.ur_stoi.items() if k.startswith('<')},
            "english_special_tokens": {k: v for k, v in self.en_stoi.items() if k.startswith('<')}
        }
