import spacy


class SpacyModel:
    _models = {}
    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._models:
            # Load model
            try:
                # Disable unnecessary component
                nlp = spacy.load(model_name)
            except OSError:
                # Download model
                from spacy.cli import download
                download(model_name)
                nlp = spacy.load(model_name)
            cls._models[model_name] = nlp
        return cls._models[model_name]