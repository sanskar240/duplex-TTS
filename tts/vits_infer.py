import torch
from .text import text_to_sequence        # was: from tts.text import text_to_sequence
from .models import SynthesizerTrn        # was: from models import SynthesizerTrn
from . import utils                       # was: import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_vits_model(model_path, config_path):
    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.eval()
    return model, hps

def infer_waveform(text, model, hps, length_scale=1.0):
    seq = text_to_sequence(text, hps.data.text_cleaners)
    x = torch.LongTensor(seq).unsqueeze(0).to(device)
    x_len = torch.LongTensor([x.size(1)]).to(device)
    with torch.no_grad():
        audio = model.infer(
            x, x_len,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=length_scale
        )[0][0, 0].cpu().numpy()
    return audio, hps.data.sampling_rate
