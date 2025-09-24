# %%
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
