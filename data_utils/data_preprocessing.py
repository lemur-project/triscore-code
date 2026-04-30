import torchaudio
import xml.etree.ElementTree as ET

class BatchModalityCollator:
    def __init__(self, processor, modality, encoder_sr, symbolic_encoding, return_index=True):
        self.processor = processor
        self.modality = modality
        self.audio_sr = encoder_sr
        self.symbolic_encoding = symbolic_encoding
        self.return_index = return_index

    def prepare_audio_inputs(self, sample):
        waveform, sr = sample
        waveform = resample_audio(waveform, sr, self.audio_sr)
        return self.processor(waveform.squeeze(0).cpu().numpy(), sampling_rate=self.audio_sr, return_tensors="pt")
    
    def prepare_image_inputs(self, sample):
        return self.processor(sample, return_tensors="pt")

    def prepare_score_inputs(self, sample):
        cl_sample = getattr(self, f"prepare_score_{self.symbolic_encoding}_inputs")(sample)
        tk_sample = self.processor(cl_sample, return_tensors="pt", truncation=True)
        return tk_sample
    
    def prepare_score_abc_inputs(self, sample):
        # replace "\n" with "<n>" 
        abc_lines = sample.split("\n")
        cleaned_sample = ""
        # remove any metadata lines, this is the processing done for the MuPT model 
        for line in abc_lines:
            if line[:2] in ['A:', 'B:', 'C:', 'D:', 'F:', 'G', 'H:', 'N:', 'O:', 'R:', 'r:', 'S:', 'T:', 'W:', 'w:', 'X:', 'Z:'] \
            or line=='\n' \
            or (line.startswith('%') and not line.startswith('%%score')):
                continue
            else:
                if "%" in line and not line.startswith('%%score'):
                    line = "%".join(line.split('%')[:-1])
                    cleaned_sample += line[:-1] + '<n>'
                else:
                    cleaned_sample += line + '<n>'

        return cleaned_sample
    
    def prepare_score_kern_inputs(self, sample):
        # remove any "!!!" lines if they exist, as these cause issues with the tokenizer
        cleaned_sample = "\n".join([line for line in sample.split("\n") if not line.startswith("!!!")])
        # remove any instrument definitions
        cleaned_sample = "\n".join(
			line for line in cleaned_sample.splitlines() if not any(
				element.startswith('*I') for element in line.split('\t')
			)
		)
        return cleaned_sample
                              
    def prepare_score_musicxml_inputs(self, sample):
        # remove any metadata tags, see all tags @ https://www.w3.org/2021/06/musicxml40/musicxml-reference/element-tree/
        root = ET.fromstring(sample)
        tags_to_remove = [
            "work", 
            "identification",
            "part-list",
            "credit",
            "movement-title",
            "movement-number"
        ]
        for tag in tags_to_remove:
            for elem in root.findall(tag):
                root.remove(elem)
        cleaned_sample = ET.tostring(root, encoding="unicode")
        return cleaned_sample

    def __call__(self, batch):
        if self.return_index:
            samples, labels, indices = zip(*batch)
        else:
            samples, labels = zip(*batch)
            indices = None
        inputs = [getattr(self, f"prepare_{self.modality}_inputs")(sample) for sample in samples]
        return inputs, labels, indices


def resample_audio(waveform, sr, target_sr):
    """
    Resample the audio waveform to the target sampling rate.
    """
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform