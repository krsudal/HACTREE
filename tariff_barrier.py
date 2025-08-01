#%%
# tariff_barrier.py

# import torch
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # 설정
# LLM_MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"
# VECTOR_DB_PATH = "vector_db/tariff_japan_cosmetics"
# EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# # 모델 로드
# print(f"💻 관세/TBT QA: Using device: {DEVICE}")
# tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     LLM_MODEL_NAME,
#     torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
# ).to(DEVICE)

# # 임베딩 및 벡터 DB 로드
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
#                                         model_kwargs={"device": DEVICE})

# try:
#     db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
# except Exception as e:
#     raise RuntimeError(f"❌ 벡터 DB 로딩 실패: {e}")

# # 질문 응답 함수
# def answer_tariff_barrier_question(question: str) -> str:
#     docs = db.similarity_search(question, k=3)
#     context = "\n".join([doc.page_content for doc in docs])

#     prompt = f"{context}\n\n질문: {question}\n답변:"

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
#     if 'token_type_ids' in inputs:
#         del inputs['token_type_ids']
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.95,
#         )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if "답변:" in response:
#         response = response.split("답변:")[-1].strip()
#     return response


# %%
import os
import pickle
import torch
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 설정
LLM_MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_ROOT = "vector_db/hs_based"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"💻 전략 QA: Using device: {DEVICE}")

# LLM & 임베딩 모델 로드
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
).to(DEVICE)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ✅ 전략 질문 응답 함수 (HS 코드 + 국가 기반)
def answer_tariff_barrier_question(hs_code: str, country: str, question: str, top_k: int = 3) -> str:
    db_path = os.path.join(VECTOR_DB_ROOT, country, hs_code)
    index_path = os.path.join(db_path, "index.faiss")
    chunks_path = os.path.join(db_path, "chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return f"[!] '{country}/{hs_code}' 벡터 DB 또는 chunk 데이터가 없습니다."

    # 1. 벡터 DB 로드
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # 2. 질문 임베딩
    query_vector = embedding_model.encode([question])
    distances, indices = index.search(query_vector, top_k)

    # 3. 관련 chunk 텍스트 추출
    selected_chunks = [chunks[i]["text"] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join(selected_chunks)

    # 4. LLM 프롬프트 구성
    prompt = f"{context}\n\n질문: {question}\n답변:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5. LLM 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "답변:" in response:
        response = response.split("답변:")[-1].strip()
    return response
