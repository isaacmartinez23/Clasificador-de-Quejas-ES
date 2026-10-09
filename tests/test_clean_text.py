from src.preprocessing.clean_text import clean_heavy, clean_heavy_batch, clean_light


def test_light_keeps_accents_case_and_emojis():
    assert clean_light("¡NO llegó el Pedido! 😡") == "¡NO llegó el Pedido! 😡"


def test_light_collapses_whitespace():
    assert clean_light("  no   llegó\n\n nunca\t") == "no llegó nunca"


def test_light_removes_urls_and_html():
    text = "Mira https://example.com/foto.jpg y www.tienda.es<br />fatal &amp; caro"
    assert clean_light(text) == "Mira y fatal & caro"


def test_light_keeps_text_that_looks_like_a_tag():
    assert clean_light("precio < 10 y > 5, <3") == "precio < 10 y > 5, <3"


def test_light_normalizes_decomposed_accents():
    assert clean_light("llegó") == "llegó"


def test_light_empty_text():
    assert clean_light("") == ""
    assert clean_light("   \n ") == ""


def test_heavy_lowercases_strips_accents_and_lemmatizes():
    tokens = clean_heavy("Los paquetes llegaron rotos").split()
    assert tokens == ["paquete", "llegar", "roto"]


def test_heavy_keeps_negations():
    assert clean_heavy("No llegó nunca").split() == ["no", "llegar", "nunca"]
    assert "sin" in clean_heavy("Llegó sin instrucciones").split()


def test_heavy_drops_function_words_numbers_punctuation_and_emojis():
    tokens = clean_heavy("El cargador de 20 W que compré es malo!!! 😡").split()
    assert tokens == ["cargador", "comprar", "malo"]


def test_heavy_keeps_complaint_words_that_spacy_lists_as_stopwords():
    tokens = clean_heavy("Llegó tarde y dejó de funcionar a los pocos días").split()
    assert {"llegar", "tarde", "dejar", "funcionar", "dia"} <= set(tokens)


def test_heavy_drops_pronoun_attached_to_verb():
    assert clean_heavy("Quiero devolverlo").split() == ["querer", "devolver"]


def test_heavy_handles_all_caps():
    assert clean_heavy("NO FUNCIONA") == clean_heavy("no funciona") == "no funcionar"


def test_heavy_empty_text():
    assert clean_heavy("") == ""


def test_heavy_batch_matches_single():
    texts = ["No llegó nunca", "Los paquetes llegaron rotos", ""]
    assert clean_heavy_batch(texts) == [clean_heavy(text) for text in texts]
