HIDDEN_DIMENSIONS = [256, 512, 1024]
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15] 
K_VALUES_TOPN = [5, 10]

MODALITIES = ["image", "audio", "score"]

DATASETS = {
    "mtd": {
        "default_dir": "data/MTD_KERN",
        "categories": ["composer", "ensemble", "instruments"],
        "map_category_names": {
            "composer": "ComposerID",
            "ensemble": "Ensemble",
            "instruments": "ThemeInstruments"
        }
    },
    "triscore": {
        "default_dir": "data/MUSCUTS",
        "categories": ["composer", "title", "instruments"],
        "map_category_names": {
            "composer": "composer",
            "title": "title",
            "instruments": "instruments"
        } 
    },
    "primus": {
        "default_dir": "data/Primus",
        "categories": ["author", "genre", "title"],
        "map_category_names": {
            "author": "Author",
            "genre": "SectionValues",
            "title": "Title"
        }
    }
}

ENCODERS = {
    # IMAGE ENCODERS
    "DiT": {
        "hidden_size": 768,
        "model_id": "microsoft/dit-base",
    },
    "ViT": {
        "hidden_size": 768,
        "model_id": "google/vit-base-patch16-224-in21k",
    },
    "PaliGemma": {
        "hidden_size": 4096,
        "model_id": "google/paligemma2-3b-pt-896",
    },
    # AUDIO ENCODERS
    "MERT": {
        "hidden_size": 1024,    
        "model_id": "m-a-p/MERT-v1-330M",
    },
    "Whisper": {
        "hidden_size": 1024,
        "model_id": "openai/whisper-medium",
    },
    "Wav2Vec2": {
        "hidden_size": 1024,
        "model_id": "ALM/wav2vec2-large-audioset",
    },
    # SYMBOLIC ENCODERS
    "RoBERTa": {
        "hidden_size": 768,
        "model_id": "roberta-base",
    },
    "GraphCodeBERT": {
        "hidden_size": 768,
        "model_id": "microsoft/graphcodebert-base",
    },
    "SentenceTransformer": {
        "hidden_size": 768,
        "model_id": "sentence-transformers/all-mpnet-base-v2",
    },
}


# Orthogonal configurations for the 9 encoder combinations
ENC_CONFIGS = {
    1: {
        "image": "ViT",
        "audio": "MERT",
        "score": "RoBERTa"
    },
    2: {
        "image": "ViT",
        "audio": "Whisper",
        "score": "GraphCodeBERT"
    },
    3: {
        "image": "ViT",
        "audio": "Wav2Vec2",
        "score": "SentenceTransformer"
    },
    4: {
        "image": "DiT",
        "audio": "MERT",
        "score": "GraphCodeBERT"
    },
    5: {
        "image": "DiT",
        "audio": "Whisper",
        "score": "SentenceTransformer"
    },
    6: {
        "image": "DiT",
        "audio": "Wav2Vec2",
        "score": "RoBERTa"
    },
    7: {
        "image": "PaliGemma",
        "audio": "MERT",
        "score": "SentenceTransformer"
    },
    8: {
        "image": "PaliGemma",
        "audio": "Whisper",
        "score": "RoBERTa"
    },
    9: {
        "image": "PaliGemma",
        "audio": "Wav2Vec2",
        "score": "GraphCodeBERT"
    },
}