import spacy


def test_spanish_spacy_model_lemmatizes():
    nlp = spacy.load("es_core_news_sm")
    lemmas = [token.lemma_ for token in nlp("Los paquetes llegaron rotos")]
    assert "paquete" in lemmas
    assert "llegar" in lemmas
