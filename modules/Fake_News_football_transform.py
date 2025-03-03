
import tensorflow as tf
LABEL_KEY = "real_news"  # Kunci untuk label
FEATURE_KEY = "tweet"  # Kunci untuk fitur teks

def transformed_name(key):
    """Menambahkan '_xf' pada nama fitur atau label untuk menunjukkan transformasi."""
    return f"{key}_xf"

def preprocessing_fn(inputs):
    """
    Melakukan preprocessing pada fitur input menjadi fitur yang telah ditransformasi.
    
    Args:
        inputs: peta dari kunci fitur ke fitur mentah.
    
    Return:
        outputs: peta dari kunci fitur ke fitur yang telah ditransformasi.
    """
    outputs = {}
    
    # Mengonversi kalimat menjadi huruf kecil untuk normalisasi
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    # Mengonversi label menjadi tipe integer
    outputs[transformed_name(LABEL_KEY)]   = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
