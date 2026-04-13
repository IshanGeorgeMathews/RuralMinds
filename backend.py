import os
import fitz
import json
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "source_folder")
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import requests


try:
    from paperqa import Docs
except ImportError:
    Docs = None

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_SENTENCES_PER_CHUNK = 5
SENTENCE_OVERLAP = 2
DEFAULT_K_RESULTS = 3
MAX_CONTEXT_LENGTH = 3000

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

# ─────────────────────────────────────────────────────────────────────────────
# Malayalam fine-tuned Whisper model path
#
# Expected files inside this directory:
#   config.json            – WhisperConfig  (encoder/decoder dims, vocab, …)
#   generation_config.json – GenerationConfig (forced_decoder_ids, suppress_tokens, …)
#   processor_config.json  – WhisperProcessor wrapper config
#   tokenizer.json         – fast-tokenizer vocab + merges
#   tokenizer_config.json  – WhisperTokenizer settings
#   model.safetensors      – fine-tuned weights
#   training_args.bin      – (optional) training metadata, not used at inference
# ─────────────────────────────────────────────────────────────────────────────
MALAYALAM_MODEL_PATH = r"C:\projects\winner\model\whisper-ml-model\content\whisper-ml-finetuned-final"

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# --- GLOBAL MODELS ---
from functools import lru_cache

_chroma_client = None
_paperqa_docs = None

@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy load embedding model with caching."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB at: {DB_PATH}")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return _chroma_client

def get_paperqa_docs():
    """Get or create PaperQA Docs instance."""
    global _paperqa_docs
    if _paperqa_docs is None:
        if Docs is None:
            raise ImportError("PaperQA not installed. Install with: pip install paperqa")
        logger.info("Initializing PaperQA")
        _paperqa_docs = Docs()
    return _paperqa_docs

def query_ollama_simple(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """Simple Ollama query without context. Used for confidence checking."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.warning(f"Ollama simple query failed: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to Ollama for confidence check")
        return None
    except Exception as e:
        logger.error(f"Error in simple Ollama query: {str(e)}")
        return None

def query_ollama_stream_simple(prompt: str, max_tokens: int = 1000):
    """Stream simple Ollama query without context. Yields chunks of text."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens
            }
        }
        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
            else:
                logger.warning(f"Ollama simple streaming failed: {response.status_code}")
                yield ""
    except Exception as e:
        logger.error(f"Error in simple Ollama streaming: {str(e)}")
        yield ""

def query_with_confidence(query: str, doc_name: str) -> Tuple[Optional[str], bool]:
    """
    Query LLM and check if it's confident about the answer.
    Returns: (answer, needs_rag)
    """
    try:
        prompt = f"""You are an AI tutor. Answer the question if you're confident in your knowledge.

If you're NOT confident, or if the question is asking about a specific document/PDF/file, respond with exactly: [NEED_CONTEXT]

Question: {query}

Answer:"""
        logger.info("🤔 Checking LLM confidence...")
        response = query_ollama_simple(prompt, max_tokens=1000)
        if not response:
            logger.info("⚡ LLM unavailable - triggering RAG")
            return None, True
        response_lower = response.lower()
        if "[need_context]" in response_lower or "need context" in response_lower or "[need context]" in response:
            logger.info("⚡ LLM not confident - triggering RAG")
            return None, True
        logger.info("✓ LLM confident - returning direct answer")
        return response.strip(), False
    except Exception as e:
        logger.error(f"Error in confidence check: {str(e)}")
        return None, True


# =============================================================================
# VOICE & ASR LAYER
# =============================================================================

# ASR model cache globals
_whisper_model     = None   # English Whisper (openai-whisper)
_hindi_asr_model   = None   # Hindi (AI4Bharat)
_malayalam_processor = None # Fine-tuned Malayalam Whisper – WhisperProcessor
_malayalam_model     = None # Fine-tuned Malayalam Whisper – WhisperForConditionalGeneration


def get_whisper_model():
    """Lazy-load openai-whisper (small) for English ASR."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("🎤 Loading Whisper (small) for English…")
        import whisper
        _whisper_model = whisper.load_model("small")
        logger.info("✓ English Whisper loaded")
    return _whisper_model


def get_hindi_asr_model():
    """Lazy-load AI4Bharat Hindi ASR pipeline."""
    global _hindi_asr_model
    if _hindi_asr_model is None:
        logger.info("🎤 Loading AI4Bharat Hindi ASR…")
        from transformers import pipeline
        _hindi_asr_model = pipeline(
            "automatic-speech-recognition",
            model="ai4bharat/indicwav2vec-hindi",
            device="cpu"
        )
        logger.info("✓ Hindi ASR loaded")
    return _hindi_asr_model


def get_malayalam_whisper():
    """
    Lazy-load the locally fine-tuned Whisper Malayalam model.

    Reads from MALAYALAM_MODEL_PATH which must contain:
        config.json, generation_config.json, model.safetensors,
        processor_config.json, tokenizer.json, tokenizer_config.json
        (training_args.bin is ignored at inference time)

    Returns:
        (WhisperProcessor, WhisperForConditionalGeneration)

    The processor bundles:
        processor.feature_extractor  →  WhisperFeatureExtractor
        processor.tokenizer          →  WhisperTokenizer
    exactly as used in the Colab fine-tuning notebook.
    """
    global _malayalam_processor, _malayalam_model

    if _malayalam_processor is None or _malayalam_model is None:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        # ── Validate model directory ──────────────────────────────────────────
        if not os.path.isdir(MALAYALAM_MODEL_PATH):
            raise FileNotFoundError(
                f"Malayalam model directory not found:\n  {MALAYALAM_MODEL_PATH}\n"
                "Please verify MALAYALAM_MODEL_PATH in backend.py."
            )

        required_files = [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        missing = [f for f in required_files
                   if not os.path.isfile(os.path.join(MALAYALAM_MODEL_PATH, f))]
        if missing:
            raise FileNotFoundError(
                f"Malayalam model directory is missing files: {missing}\n"
                f"Directory: {MALAYALAM_MODEL_PATH}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"🎤 Loading fine-tuned Malayalam Whisper…\n"
            f"   Path   : {MALAYALAM_MODEL_PATH}\n"
            f"   Device : {device}"
        )

        # ── Load processor (feature_extractor + tokenizer bundled) ────────────
        # local_files_only=True ensures we never accidentally hit the HF hub.
        _malayalam_processor = WhisperProcessor.from_pretrained(
            MALAYALAM_MODEL_PATH,
            local_files_only=True,
        )

        # ── Load model weights ────────────────────────────────────────────────
        # Force FP32 – FP16 is not supported on CPU and causes NaN outputs.
        _malayalam_model = WhisperForConditionalGeneration.from_pretrained(
            MALAYALAM_MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)

        _malayalam_model.eval()

        # ── Patch generation_config so it always transcribes Malayalam ────────
        # The checkpoint's generation_config may have forced_decoder_ids that
        # point to English; override them here to be safe.
        _malayalam_model.generation_config.forced_decoder_ids = None
        _malayalam_model.generation_config.suppress_tokens    = []

        logger.info("✓ Fine-tuned Malayalam Whisper loaded successfully")

    return _malayalam_processor, _malayalam_model


# ---------------------------------------------------------------------------
# Per-language transcription helpers
# ---------------------------------------------------------------------------

def transcribe_english(audio_path: str) -> str:
    """Transcribe English audio using openai-whisper."""
    model = get_whisper_model()
    result = model.transcribe(audio_path, language="en")
    return result.get("text", "").strip()


def transcribe_hindi(audio_path: str) -> str:
    """Transcribe Hindi audio using AI4Bharat model."""
    model = get_hindi_asr_model()
    result = model(audio_path)
    if isinstance(result, list):
        return result[0].get("text", "").strip()
    return result.get("text", "").strip()


def transcribe_malayalam(audio_path: str) -> str:
    """
    Transcribe Malayalam audio using the locally fine-tuned Whisper model.
    Replicates the working Colab pipeline exactly:
        processor.feature_extractor → .input_features
        processor.get_decoder_prompt_ids(language="ml", task="transcribe")
        model.generate(...)
        processor.tokenizer.decode(...)
    """
    import torch
    import librosa
    import numpy as np

    processor, model = get_malayalam_whisper()
    device = next(model.parameters()).device

    # ── 1. Load & resample to 16 kHz (exactly as Colab) ──────────────────────
    logger.info(f"  Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = np.asarray(audio, dtype=np.float32)
    logger.info(f"  Audio shape={audio.shape}, sr={sr}")

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    # Mirrors Colab exactly: processor.feature_extractor(...).input_features
    # The tokenizer_config.json fix (extra_special_tokens: {} ) means
    # processor loads cleanly now, so we can call it the same way as Colab.
    feat_out = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    # Defensive: handle BatchFeature, dict, or rare list return
    if hasattr(feat_out, "input_features"):
        input_features = feat_out.input_features.to(device)
    elif isinstance(feat_out, dict):
        input_features = feat_out["input_features"].to(device)
    elif isinstance(feat_out, list):
        input_features = torch.tensor(np.array(feat_out[0])).unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unexpected feature extractor output type: {type(feat_out)}")
    logger.info(f"  input_features shape: {input_features.shape}")

    # ── 3. Forced decoder ids — language="ml" exactly as Colab ───────────────
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ml",
        task="transcribe"
    )
    logger.info(f"  forced_decoder_ids: {forced_decoder_ids}")

    # ── 4. Generate ───────────────────────────────────────────────────────────
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=200,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    # ── 5. Decode — exactly as Colab ──────────────────────────────────────────
    text = processor.tokenizer.decode(
        predicted_ids[0],
        skip_special_tokens=True
    )

    logger.info(f"🔥 Fine-Tuned Malayalam Output: {text}")
    return text.strip()


# ---------------------------------------------------------------------------
# Unified transcription entry point
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, language_code: str = "en") -> tuple:
    """
    Transcribe audio using language-specific ASR models.

    Args:
        audio_path    : Path to audio file (WAV recommended; 16 kHz mono).
        language_code : 'en' | 'hi' | 'ml'

    Returns:
        (success: bool, message: str, data: dict | None)
        data = {'text': str} on success, None on failure.
    """
    try:
        logger.info(f"🎤 Transcribing [{language_code.upper()}]: {audio_path}")

        if language_code == "ml":
            # ── Malayalam: ALWAYS use the local fine-tuned model ──────────────
            text = transcribe_malayalam(audio_path)
            if not text:
                return False, "Voice not clear or no speech detected. Please try again.", None
            return True, "Transcription successful", {"text": text}

        elif language_code == "hi":
            text = transcribe_hindi(audio_path)

        elif language_code == "en":
            text = transcribe_english(audio_path)

        else:
            logger.warning(f"Unknown language '{language_code}', falling back to English Whisper")
            text = transcribe_english(audio_path)

        if not text:
            return False, "Voice not clear. Please try again.", None

        logger.info(f"🎤 Transcription result: {text}")
        return True, "Transcription successful", {"text": text}

    except FileNotFoundError as e:
        # Surface missing-model-dir errors clearly
        logger.error(f"Model not found: {str(e)}")
        return False, f"Model not found: {str(e)}", None
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return False, f"Transcription failed: {str(e)}", None


def normalize_query(query: str) -> str:
    """Deprecated: kept for backward compat. Detects lang and translates to EN."""
    lang = detect_language(query)
    if lang != 'en':
        return translate_to_english(query, lang)
    return query


# =============================================================================
# LANGUAGE DETECTION & TRANSLATION
# =============================================================================

_translation_models = {}

def detect_language(text: str) -> str:
    """
    Detect language using langdetect + Romanized keyword heuristics.
    Returns: 'en' | 'hi' | 'ml'
    """
    try:
        from langdetect import detect

        indic_keywords = {
            'hi': ['bhai', 'kya', 'hai', 'kaise', 'kyun', 'aur', 'yeh', 'woh'],
            'ml': ['entha', 'engane', 'evide', 'cheyyane', 'und', 'illa', 'enth']
        }

        text_lower = text.lower()
        for lang, keywords in indic_keywords.items():
            if any(word in text_lower for word in keywords):
                logger.info(f"🔍 Detected Romanized {lang.upper()}")
                return lang

        detected = detect(text)
        logger.info(f"🔍 langdetect: {detected}")
        return detected if detected in ['hi', 'ml', 'en'] else 'en'

    except Exception as e:
        logger.warning(f"Language detection failed ({e}), defaulting to English")
        return 'en'


def get_translation_model(source_lang: str, target_lang: str):
    """Load and cache MarianMT translation model."""
    global _translation_models
    model_key = f"{source_lang}-{target_lang}"

    if model_key not in _translation_models:
        from transformers import MarianMTModel, MarianTokenizer

        model_map = {
            'hi-en': 'Helsinki-NLP/opus-mt-hi-en',
            'ml-en': 'Helsinki-NLP/opus-mt-ml-en',
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'en-ml': 'Helsinki-NLP/opus-mt-en-ml'
        }

        if model_key not in model_map:
            logger.warning(f"No translation model for {model_key}")
            return None, None

        model_name = model_map[model_key]
        logger.info(f"📥 Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _translation_models[model_key] = (tokenizer, model)
        logger.info(f"✓ Translation model loaded: {model_key}")

    return _translation_models[model_key]


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text from source_lang to English using MarianMT."""
    if source_lang == 'en':
        return text
    try:
        tokenizer, model = get_translation_model(source_lang, 'en')
        if not tokenizer or not model:
            return text
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        logger.info(f"🌍 Translated ({source_lang}→en): '{text}' → '{result}'")
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate text from English to target_lang using MarianMT."""
    if target_lang == 'en':
        return text
    try:
        tokenizer, model = get_translation_model('en', target_lang)
        if not tokenizer or not model:
            return text
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        logger.info(f"🌍 Translated (en→{target_lang}): '{text}' → '{result}'")
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text


def is_document_query(query: str) -> bool:
    """Keyword-based check: is this query about a specific document?"""
    if not isinstance(query, str):
        logger.warning(f"is_document_query received non-string: {type(query)}")
        return False

    doc_keywords = [
        'pdf', 'document', 'file', 'chapter', 'page', 'section',
        'this doc', 'the doc', 'this pdf', 'the pdf', 'this file'
    ]
    is_doc = any(kw in query.lower() for kw in doc_keywords)
    logger.info("📄 Document-specific query" if is_doc else "💬 General query")
    return is_doc


# =============================================================================
# PDF PROCESSING
# =============================================================================

def sanitize_collection_name(filename: str) -> str:
    """Sanitize filename for ChromaDB collection name."""
    name = Path(filename).stem
    name = "".join(c if c.isalnum() or c in '._-' else '_' for c in name)
    name = name.lstrip("0123456789_-")
    if len(name) < 3:
        name = f"doc_{name}"
    return name[:63].lower()


def extract_pdf(file_stream) -> str:
    """Extract text from PDF file stream."""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(text)
        doc.close()
        if not text_parts:
            logger.warning("No text extracted from PDF")
            return ""
        return " ".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()


def sentence_based_chunking(
    text: str,
    max_sentences: int = MAX_SENTENCES_PER_CHUNK,
    overlap: int = SENTENCE_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks based on sentences."""
    if not text or not text.strip():
        return []
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing: {str(e)}")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return []
    chunks = []
    overlap = min(overlap, max_sentences - 1)
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# =============================================================================
# HYBRID RAG SYSTEM
# =============================================================================

UPLOADED_PDFS_PATH = "uploaded_pdfs"
os.makedirs(UPLOADED_PDFS_PATH, exist_ok=True)

def process_and_save_pdf(uploaded_file) -> Tuple[bool, str]:
    """Process PDF with ChromaDB and save globally to SQLite & File System."""
    try:
        raw_text = extract_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = sentence_based_chunking(cleaned_text)

        if not chunks:
            return False, "Could not extract any text from the PDF."

        embed_model = get_embedding_model()
        embeddings = embed_model.encode(chunks, show_progress_bar=False)

        doc_name = sanitize_collection_name(uploaded_file.name)
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=doc_name)

        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{
                "type": "pdf",
                "chunk_id": i,
                "source": uploaded_file.name,
                "created_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )

        # Permanent Document Storage
        permanent_path = os.path.join(UPLOADED_PDFS_PATH, f"{doc_name}.pdf")
        with open(permanent_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        try:
            from database import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                INSERT OR IGNORE INTO documents (filename, upload_path, uploaded_by, uploaded_at)
                VALUES (?, ?, ?, ?)
            ''', (doc_name, permanent_path, "teacher", datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not log to documents SQLite table: {str(e)}")

        # Optional PaperQA handling
        try:
            docs = get_paperqa_docs()
            docs.add(permanent_path, docname=doc_name)
            logger.info(f"Added {doc_name} to PaperQA")
        except Exception as e:
            logger.warning(f"Could not add to PaperQA: {str(e)}")

        return True, f"Successfully processed '{uploaded_file.name}' ({len(chunks)} chunks)."

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False, f"Error: {str(e)}"


# =============================================================================
# OLLAMA QUERY FUNCTIONS
# =============================================================================

def preload_ollama_model():
    """Warm up Ollama so the model is in RAM before first user query."""
    try:
        logger.info(f"Warming up Ollama model: {OLLAMA_MODEL}…")
        warmup_payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Hello",
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 5}
        }
        response = requests.post(OLLAMA_API_URL, json=warmup_payload, timeout=30)
        if response.status_code == 200:
            logger.info(f"✓ Ollama model {OLLAMA_MODEL} ready")
            return True
        logger.warning(f"Ollama warm-up status: {response.status_code}")
        return False
    except Exception as e:
        logger.warning(f"Could not preload Ollama: {str(e)}")
        return False


def query_ollama_stream(context: str, query: str):
    """Stream Ollama response with context. Yields text chunks."""
    try:
        system_prompt = """
You are an AI Tutor.

Rules:
0. You can get questions in English, Malayalam or Hindi.
1. Answer the question directly and simply.
2. Do NOT guess the user's intention beyond the question.
3. Do NOT mention language detection.
4. Do NOT translate unless explicitly asked.
5. If the question is informal or partially in another language, interpret it as a simple academic question.
6. Never explain what language the user used.
7. Do not add extra commentary.
8. Keep answers concise and correct.
9. If you don't understand some words, don't ask about them in brackets.
10. "me farak" means difference between.
11. "kya hai" means what do you mean by that topic.
12. "enthanu" also means what do you mean by that topic.
13. Always check if the language is in Malayalam or Hindi (and same with words).
14. User will never ask you anything in Tamil; if you think it's Tamil, the language is Malayalam.
If the user asks in mixed language (e.g., Malayalam/Hindi written in English),
interpret the meaning and reply in that language.
"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            "stream": True,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": 150}
        }

        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
            else:
                yield "Error: Could not connect to AI model."

    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {str(e)}")
        yield f"Error: {str(e)}"


def query_ollama(context: str, query: str) -> Optional[str]:
    """Blocking Ollama query with context."""
    try:
        system_prompt = """You are a helpful AI tutor.
Answer clearly using only the provided context.
Use short structured explanations."""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": 150}
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get('response', '')
        logger.warning(f"Ollama status: {response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {str(e)}")
        return None


def generate_answer_from_context(
    retrieved_chunks: List[str],
    query: str,
    max_length: int = MAX_CONTEXT_LENGTH
) -> str:
    """Generate answer from retrieved chunks (blocking)."""
    if not retrieved_chunks:
        return "No relevant information found in the document."

    context = ""
    for chunk in retrieved_chunks:
        if len(context) + len(chunk) < max_length:
            context += chunk + "\n\n"
        else:
            break
    context = context.strip()

    if not context:
        return "Retrieved context too long to process."

    try:
        response = query_ollama(context, query)
        if response:
            return response
    except Exception as e:
        logger.warning(f"Ollama generation failed: {str(e)}")

    return f"""Based on the document, here is the relevant information:

QUESTION: {query}

RELEVANT EXCERPTS:
{context}

(Note: AI generation unavailable - showing direct extracts)"""


def generate_answer_from_context_stream(
    retrieved_chunks: List[str],
    query: str,
    max_length: int = MAX_CONTEXT_LENGTH
):
    """Generate answer from retrieved chunks (streaming)."""
    if not retrieved_chunks:
        yield "No relevant information found in the document."
        return

    context = ""
    for chunk in retrieved_chunks:
        if len(context) + len(chunk) < max_length:
            context += chunk + "\n\n"
        else:
            break
    context = context.strip()

    if not context:
        yield "Retrieved context too long to process."
        return

    try:
        for chunk in query_ollama_stream(context, query):
            yield chunk
        return
    except Exception as e:
        logger.warning(f"Ollama generation failed: {str(e)}")

    yield f"""Based on the document, here is the relevant information:

QUESTION: {query}

RELEVANT EXCERPTS:
{context}

(Note: AI generation unavailable - showing direct extracts)"""


# =============================================================================
# HYBRID QUERY SYSTEM
# =============================================================================

def query_saved_document_hybrid(
    doc_name: str,
    query: str,
    k: int = DEFAULT_K_RESULTS
) -> Tuple[str, List[str]]:
    """Confidence-based hybrid: try LLM first, fall back to RAG."""
    try:
        llm_answer, needs_rag = query_with_confidence(query, doc_name)
        if not needs_rag and llm_answer:
            return llm_answer, []
    except Exception as e:
        logger.warning(f"Confidence check failed: {str(e)}, proceeding to RAG")

    logger.info("📚 Running full RAG pipeline…")
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=doc_name)
    except ValueError:
        return f"Error: Document '{doc_name}' not found.", []
    except Exception as e:
        return f"Error accessing database: {str(e)}", []

    try:
        embed_model = get_embedding_model()
        query_emb = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=k)
        retrieved_chunks = results['documents'][0] if results['documents'] else []

        if not retrieved_chunks:
            return "No relevant information found for this query.", []

        answer = generate_answer_from_context(retrieved_chunks, query)
        return answer, retrieved_chunks

    except Exception as e:
        logger.error(f"Error querying: {str(e)}")
        return f"Error generating answer: {str(e)}", []


def query_saved_document_stream(
    doc_name: str,
    query: str,
    k: int = DEFAULT_K_RESULTS,
    forced_language: str = None
):
    """
    Streaming query pipeline.

    1. Detect/force language
    2. Translate to English if needed
    3. Keyword check: document-specific → RAG | general → direct LLM
    4. Stream answer chunks; final yield is {'sources': [...]}
    """
    if forced_language:
        original_lang = forced_language
        logger.info(f"🔒 Forced language: {forced_language}")
    else:
        original_lang = detect_language(query)
        logger.info(f"🔍 Auto-detected language: {original_lang}")

    english_query = query
    if original_lang != 'en':
        english_query = translate_to_english(query, original_lang)
        logger.info(f"🔄 Translated query: {english_query}")

    if is_document_query(english_query):
        logger.info("📚 Running RAG pipeline…")
        try:
            client = get_chroma_client()
            collection = client.get_collection(name=doc_name)
            embed_model = get_embedding_model()
            query_emb = embed_model.encode([english_query]).tolist()
            results = collection.query(query_embeddings=query_emb, n_results=k)
            retrieved_chunks = results['documents'][0] if results['documents'] else []

            if not retrieved_chunks:
                yield "No relevant information found for this query."
                yield {'sources': []}
                return

            for chunk in generate_answer_from_context_stream(retrieved_chunks, english_query):
                yield chunk
            yield {'sources': retrieved_chunks}

        except Exception as e:
            yield f"Error during RAG: {str(e)}"
            yield {'sources': []}

    else:
        logger.info("💬 Using general LLM…")
        try:
            prompt = f"""You are an AI Tutor.

Rules:
0. You can get questions in English, Malayalam or Hindi.
1. Answer the question directly and simply.
2. Do NOT guess the user's intention beyond the question.
3. Do NOT mention language detection.
4. Do NOT translate unless explicitly asked.
5. If the question is informal or partially in another language, interpret it as a simple academic question.
6. Never explain what language the user used.
7. Do not add extra commentary.
8. Keep answers concise and correct.

Question: {english_query}

Answer:"""
            for chunk in query_ollama_stream_simple(prompt, max_tokens=1000):
                yield chunk
            yield {'sources': []}

        except Exception as e:
            yield f"Error during LLM query: {str(e)}"
            yield {'sources': []}


# =============================================================================
# COMPATIBILITY WRAPPERS
# =============================================================================

def query_saved_document(doc_name: str, query: str, k: int = DEFAULT_K_RESULTS) -> Tuple[str, List[str]]:
    """Backward compatible wrapper."""
    return query_saved_document_hybrid(doc_name, query, k)


def get_available_documents() -> List[str]:
    """Get list of all indexed documents."""
    client = get_chroma_client()
    collections = client.list_collections()
    return sorted([col.name for col in collections])


def get_document_path(doc_name: str) -> Optional[str]:
    """Get the physical file path of a document."""
    try:
        from database import get_db_connection
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT upload_path FROM documents WHERE filename = ?', (doc_name,))
        row = c.fetchone()
        conn.close()
        if row and os.path.exists(row['upload_path']):
            return row['upload_path']
        return None
    except Exception:
        return None

def delete_document(doc_name: str) -> Tuple[bool, str]:
    """Delete a document from ChromaDB and SQLite."""
    try:
        client = get_chroma_client()
        try:
            client.get_collection(name=doc_name)
        except Exception:
            pass
        
        try:
            client.delete_collection(name=doc_name)
            logger.info(f"Deleted collection: {doc_name}")
        except:
            pass
            
        try:
            from database import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT upload_path FROM documents WHERE filename = ?', (doc_name,))
            row = c.fetchone()
            
            if row and os.path.exists(row['upload_path']):
                os.remove(row['upload_path'])
                
            c.execute('DELETE FROM documents WHERE filename = ?', (doc_name,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to clear SQL reference: {e}")
            
        return True, f"Document '{doc_name}' deleted successfully."
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False, f"Error deleting document: {str(e)}"


def get_document_stats(doc_name: str) -> Optional[Dict]:
    """Get chunk count and type for a document."""
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=doc_name)
        count = collection.count()
        sample = collection.get(limit=1, include=['metadatas'])
        doc_type = "pdf"
        if sample and sample['metadatas']:
            doc_type = sample['metadatas'][0].get('type', 'pdf')
        return {"name": doc_name, "chunk_count": count, "type": doc_type}
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return None


def rebuild_database() -> Tuple[bool, str]:
    """Rebuild entire ChromaDB from source files."""
    try:
        logger.info("♻️ Starting database rebuild…")
        pdf_count = 0
        if os.path.exists(SOURCE_FOLDER):
            for filename in os.listdir(SOURCE_FOLDER):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(SOURCE_FOLDER, filename)
                    with open(file_path, 'rb') as f:
                        from io import BytesIO

                        class NamedBytesIO(BytesIO):
                            def __init__(self, content, name):
                                super().__init__(content)
                                self.name = name

                        file_obj = NamedBytesIO(f.read(), filename)
                        success, msg = process_and_save_pdf(file_obj)
                        if success:
                            pdf_count += 1

        video_count = 0
        if os.path.exists(VIDEO_STORAGE_PATH):
            for filename in os.listdir(VIDEO_STORAGE_PATH):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    video_name = Path(filename).stem
                    caption_data = load_caption_file(video_name)
                    if caption_data:
                        success, msg = process_video_captions(video_name, caption_data['full_text'])
                        if success:
                            video_count += 1

        logger.info(f"✅ Rebuild complete. PDFs: {pdf_count}, Videos: {video_count}")
        return True, f"Rebuild complete. Indexed {pdf_count} documents and {video_count} video captions."

    except Exception as e:
        logger.error(f"Database rebuild failed: {str(e)}")
        return False, f"Rebuild failed: {str(e)}"


# =============================================================================
# VIDEO FUNCTIONS
# =============================================================================

VIDEO_STORAGE_PATH = "static/videos"
CAPTIONS_STORAGE_PATH = "captions"

os.makedirs(VIDEO_STORAGE_PATH, exist_ok=True)
os.makedirs(CAPTIONS_STORAGE_PATH, exist_ok=True)


def save_video(uploaded_file) -> Tuple[bool, str, Optional[str]]:
    """Save an uploaded video file and log to SQLite DBMS."""
    try:
        original_name = uploaded_file.name
        safe_name = sanitize_collection_name(original_name)
        extension = Path(original_name).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}{extension}"
        video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
        
        uploaded_file.seek(0)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
            
        try:
            from database import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO videos (filename, name, video_path, has_captions, uploaded_by, uploaded_at)
                VALUES (?, ?, ?, 0, ?, ?)
            ''', (filename, Path(original_name).stem, video_path, "teacher", datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to log video to SQLite: {db_err}")
            
        logger.info(f"Video saved: {video_path}")
        return True, f"Video '{original_name}' saved successfully.", video_path
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return False, f"Error saving video: {str(e)}", None


def save_caption_file(video_name: str, caption_text: str, timestamps: Optional[List[dict]] = None) -> Tuple[bool, str]:
    """Save caption text for a video as JSON and update SQLite."""
    try:
        safe_name = sanitize_collection_name(video_name)
        caption_filename = f"{safe_name}_captions.json"
        caption_path = os.path.join(CAPTIONS_STORAGE_PATH, caption_filename)
        caption_data = {
            "video_name": video_name,
            "created_at": datetime.now().isoformat(),
            "full_text": caption_text,
            "timestamps": timestamps or [],
            "word_count": len(caption_text.split())
        }
        with open(caption_path, "w", encoding="utf-8") as f:
            json.dump(caption_data, f, indent=2, ensure_ascii=False)
            
        try:
            from database import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                UPDATE videos SET caption_path = ?, has_captions = 1 
                WHERE name = ?
            ''', (caption_path, video_name))
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to log caption to SQLite: {db_err}")
            
        logger.info(f"Caption saved: {caption_path}")
        return True, f"Caption saved for '{video_name}'."
    except Exception as e:
        logger.error(f"Error saving caption: {str(e)}")
        return False, f"Error saving caption: {str(e)}"


def load_caption_file(video_name: str) -> Optional[dict]:
    """Load caption data for a video."""
    try:
        safe_name = sanitize_collection_name(video_name)
        caption_filename = f"{safe_name}_captions.json"
        caption_path = os.path.join(CAPTIONS_STORAGE_PATH, caption_filename)
        if not os.path.exists(caption_path):
            return None
        with open(caption_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading caption: {str(e)}")
        return None


def process_video_captions(video_name: str, caption_text: str) -> Tuple[bool, str]:
    """Process and index video captions in ChromaDB."""
    try:
        cleaned_text = clean_text(caption_text)
        chunks = sentence_based_chunking(cleaned_text)
        if not chunks:
            return False, "Could not create chunks from caption text."
        embed_model = get_embedding_model()
        embeddings = embed_model.encode(chunks, show_progress_bar=False)
        doc_name = sanitize_collection_name(video_name)
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=doc_name)
        ids = [f"{doc_name}_caption_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{
                "type": "video_caption",
                "chunk_id": i,
                "source": video_name,
                "created_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )
        return True, f"Captions for '{video_name}' indexed successfully ({len(chunks)} chunks)."
    except Exception as e:
        logger.error(f"Error processing video captions: {str(e)}")
        return False, f"Error: {str(e)}"


def get_available_videos() -> List[dict]:
    """Get list of available videos from SQLite."""
    try:
        from database import get_db_connection
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM videos')
        rows = c.fetchall()
        
        videos = []
        for row in rows:
            if os.path.exists(row['video_path']):
                videos.append({
                    "filename": row['filename'],
                    "name": row['name'],
                    "path": row['video_path'],
                    "has_captions": bool(row['has_captions']),
                    "caption_data": load_caption_file(row['name']),
                    "size_mb": round(os.path.getsize(row['video_path']) / (1024 * 1024), 2)
                })
        conn.close()
        return sorted(videos, key=lambda x: x['filename'])
    except Exception as e:
        logger.error(f"Error fetching videos from DB: {str(e)}")
        return []


def delete_video(video_name: str) -> Tuple[bool, str]:
    """Delete a video and all its associated files from Disk and SQLite."""
    try:
        deleted_items = []
        
        try:
            from database import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute('SELECT video_path, caption_path FROM videos WHERE name = ?', (video_name,))
            row = c.fetchone()
            
            if row:
                if row['video_path'] and os.path.exists(row['video_path']):
                    os.remove(row['video_path'])
                    deleted_items.append("video file")
                if row['caption_path'] and os.path.exists(row['caption_path']):
                    os.remove(row['caption_path'])
                    deleted_items.append("caption file")
                    
            c.execute('DELETE FROM videos WHERE name = ?', (video_name,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed SQL Video Deletion cleanup: {e}")
            
        # Fallback disk scan for stragglers
        if not deleted_items:
            if os.path.exists(VIDEO_STORAGE_PATH):
                for filename in os.listdir(VIDEO_STORAGE_PATH):
                    if Path(filename).stem == video_name:
                        video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
                        os.remove(video_path)
                        deleted_items.append(f"video file: {filename}")
            safe_name = sanitize_collection_name(video_name)
            caption_path = os.path.join(CAPTIONS_STORAGE_PATH, f"{safe_name}_captions.json")
            if os.path.exists(caption_path):
                os.remove(caption_path)
                deleted_items.append("caption file")
                
        try:
            client = get_chroma_client()
            client.delete_collection(name=sanitize_collection_name(video_name))
            deleted_items.append("database collection")
        except Exception as e:
            logger.info(f"No ChromaDB collection for {video_name}: {str(e)}")
            
        if deleted_items:
            return True, f"Deleted {', '.join(deleted_items)} for '{video_name}'."
        return False, f"No files found for video '{video_name}'."
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return False, f"Error deleting video: {str(e)}"


# =============================================================================
# AUDIO / VIDEO CAPTION GENERATION
# =============================================================================

def extract_audio_from_video(video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Extract mono 16 kHz WAV audio from video using ffmpeg."""
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        command = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-y',
            audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}", None
        if not os.path.exists(audio_path):
            return False, "Audio file was not created", None
        return True, "Audio extracted successfully", audio_path
    except FileNotFoundError:
        return False, "FFmpeg not found. Install from: https://ffmpeg.org/download.html", None
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False, f"Error: {str(e)}", None


def generate_captions_from_video(video_path: str, video_name: str, language: str = "en") -> Tuple[bool, str, Optional[str]]:
    """Generate captions from video using the appropriate ASR model."""
    try:
        logger.info(f"Extracting audio from: {video_name}")
        success, message, audio_path = extract_audio_from_video(video_path)
        if not success:
            return False, message, None

        logger.info(f"Transcribing: {video_name}")
        success, message, transcription_data = transcribe_audio(audio_path, language)
        if not success:
            return False, message, None

        caption_text = transcription_data['text']
        timestamps = []
        for segment in transcription_data.get('segments', []):
            timestamps.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '')
            })

        save_caption_file(video_name, caption_text, timestamps)
        word_count = len(caption_text.split())
        return True, f"Captions generated! ({word_count} words)", caption_text

    except Exception as e:
        logger.error(f"Error generating captions: {str(e)}")
        return False, f"Error: {str(e)}", None


def translate_text(text: str, target_language: str = "es") -> Tuple[bool, str, Optional[str]]:
    """Translate text to target language."""
    try:
        if target_language == "en":
            return True, "No translation needed", text
        from transformers import pipeline
        lang_models = {
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "ml": "Helsinki-NLP/opus-mt-en-ml",
        }
        if target_language not in lang_models:
            return False, f"Language '{target_language}' not supported", None
        model_name = lang_models[target_language]
        logger.info(f"Loading translation model: {model_name}")
        translation_pipe = pipeline("translation", model=model_name)
        max_length = 500
        sentences = sent_tokenize(text)
        translated_chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length and current_chunk:
                chunk_text = " ".join(current_chunk)
                result = translation_pipe(chunk_text, max_length=512)
                translated_chunks.append(result[0]['translation_text'])
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            result = translation_pipe(chunk_text, max_length=512)
            translated_chunks.append(result[0]['translation_text'])
        translated_text = " ".join(translated_chunks)
        return True, f"Translation to '{target_language}' completed", translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return False, f"Error: {str(e)}", None