import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
# Override to http://localhost:11434/v1 for Ollama local testing
QWEN_API_URL = os.getenv("QWEN_API_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-max")
INTENT_MODEL = os.getenv("INTENT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")

TOP_K_SYNTHEA = int(os.getenv("TOP_K_SYNTHEA", "5"))
TOP_K_MTSAMPLES = int(os.getenv("TOP_K_MTSAMPLES", "3"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "25"))

CHROMA_PATH = os.getenv("CHROMA_PATH", "../chroma_db")

# Embeddings handled by sentence_transformers (see services/embedder.py) — no separate client needed
llm_client = OpenAI(base_url=QWEN_API_URL, api_key=QWEN_API_KEY or "ollama")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

synthea_col = chroma_client.get_collection("synthea_structured")
mtsamples_col = chroma_client.get_collection("mtsamples_knowledge")
patient_index_col = chroma_client.get_collection("patient_index")
chat_history_col = chroma_client.get_or_create_collection("chat_history")
activity_log_col = chroma_client.get_or_create_collection("activity_log")
