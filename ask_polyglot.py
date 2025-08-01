# %%
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# import time

# # ✅ 설정
# VECTOR_DB_PATH = "vector_db/export_faiss"
# EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
# LLM_MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"

# # ✅ 디바이스 설정
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"[📡] Using device: {DEVICE}")

# # ✅ 임베딩 모델 로드
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# # ✅ FAISS DB 로드
# db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# # ✅ LLM 로드
# tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16)
# model.to(DEVICE)

# # ✅ 질문 입력
# question = input("💬 질문을 입력하세요: ")

# # ✅ 검색
# start_time = time.time()
# docs = db.similarity_search(question, k=3)
# retrieved_context = "\n\n".join([doc.page_content for doc in docs])
# print("\n🔍 검색된 문서 요약:")
# print(retrieved_context[:500] + ("..." if len(retrieved_context) > 500 else ""))

# # ✅ 프롬프트 구성
# prompt = f"""
# 아래 문서를 참고하여 질문에 답하십시오.

# 문서:
# {retrieved_context}

# 질문: {question}
# 답변:
# """

# # ✅ 토큰화 및 생성
# inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
# inputs.pop("token_type_ids", None)  # <-- 이 줄 추가
# inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7,
#     )

# print("\n🧠 Polyglot-ko 응답:")
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# print(f"\n⏱️ 응답 생성 시간: {round(time.time() - start_time, 2)}초")

# %%
# !pip install tf-keras

# %%
#ask_polyglot
# ask_polyglot.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 🔧 설정
LLM_MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"
VECTOR_DB_PATH = "vector_db/export_faiss"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 🧠 모델 로드
print(f"💻 Using device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
)
model.to(DEVICE)

# 🧠 벡터 DB 및 임베딩 로드
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                        model_kwargs={"device": DEVICE})
db = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# ❓ 질문에 답변하는 함수
def answer_question_from_vectorstore(question: str) -> str:
    # 1. 유사 문서 검색
    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 2. 프롬프트 구성
    prompt = f"{context}\n\n질문: {question}\n답변:"

    # 3. 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    # 4. token_type_ids 제거 (GPT류 모델 비호환)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5. 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

    # 6. 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 선택: 질문 이후만 추출하거나 "답변:" 이후만 추출 (간단한 정제)
    if "답변:" in response:
        response = response.split("답변:")[-1].strip()
    return response

# %%
