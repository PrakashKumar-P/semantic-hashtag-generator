import pytest
import pandas as pd
from sentence_transformers import SentenceTransformer
from trendsnap_sbert import load_model, load_data, generate_hashtags



# 1️ Model Loading Test

def test_model_loading():
    """Ensure SentenceTransformer model loads properly."""
    model = load_model()
    assert isinstance(model, SentenceTransformer)
    assert model is not None



# 2️ Dataset Loading Test

def test_data_loading():
    """Check dataset is loaded and processed correctly."""
    df, captions = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Dataset is empty!"
    assert 'caption' in df.columns
    assert 'hashtags' in df.columns
    assert isinstance(captions, list)
    assert len(captions) > 0



# 3️ Hashtag Generation Test

def test_hashtag_generation_valid_input():
    """Test hashtag generation for a normal caption."""
    error, hashtags = generate_hashtags("Exploring the beach with friends!")
    assert error is None
    assert isinstance(hashtags, list)
    assert len(hashtags) > 0



# 4️ Empty Caption Handling

def test_hashtag_generation_empty_input():
    """Test if empty captions are handled gracefully."""
    error, hashtags = generate_hashtags(" ")
    assert error is not None
    assert hashtags == []



# 5️ Model Output Consistency

def test_model_consistency():
    """Ensure model gives consistent embeddings for the same caption."""
    model = load_model()
    emb1 = model.encode("A beautiful sunset at the beach")
    emb2 = model.encode("A beautiful sunset at the beach")
    assert (emb1 == emb2).all(), "Embeddings are inconsistent for the same input!"



# 6️ Performance Test (Optional)

def test_hashtag_generation_speed(benchmark):
    """Check hashtag generation speed using pytest-benchmark plugin."""
    benchmark(lambda: generate_hashtags("Testing the speed of hashtag generation"))
