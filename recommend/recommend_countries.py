# recommend/recommend_countries.py
#%%
# import os
# import json
# from typing import List, Tuple
# from recommend.scorer import calculate_score
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle

# # 환경 설정
# VECTOR_DB_PATH = "vector_db/hs_based"
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# model = SentenceTransformer(EMBEDDING_MODEL)

# def recommend_top_countries(hs_code: str, top_n: int = 3) -> List[Tuple[str, float]]:
#     """
#     주어진 HS 코드에 대해 각국 벡터 DB에서 chunk를 검색하고,
#     스코어 계산을 통해 유망 수출국 Top-N을 반환합니다.
    
#     Returns:
#         List of tuples: [(country, score), ...]
#     """

#     country_scores = []

#     for country in os.listdir(VECTOR_DB_PATH):
#         country_path = os.path.join(VECTOR_DB_PATH, country, hs_code)
#         if not os.path.exists(country_path):
#             continue

#         try:
#             # FAISS 벡터와 메타정보 로딩
#             with open(os.path.join(country_path, "index.faiss"), "rb") as f:
#                 index = faiss.read_index(faiss.read_index_binary(f.read()))
#             with open(os.path.join(country_path, "chunks.pkl"), "rb") as f:
#                 chunks = pickle.load(f)  # List of dict: {"text", "import_val", "tariff", "tbt_level"}

#             # 임시 평균 기반 스코어 계산
#             total_score = 0
#             valid_chunks = 0
#             for chunk in chunks:
#                 try:
#                     score = calculate_score(
#                         import_val=chunk.get("import_val", 0.0),
#                         tariff=chunk.get("tariff", 0.3),  # 0.3 = 예시 평균 관세율
#                         tbt_level=chunk.get("tbt_level", 0.5)  # 0.5 = 예시 중간 복잡도
#                     )
#                     total_score += score
#                     valid_chunks += 1
#                 except Exception:
#                     continue

#             if valid_chunks > 0:
#                 avg_score = total_score / valid_chunks
#                 country_scores.append((country, avg_score))

#         except Exception as e:
#             print(f"[!] {country} 처리 실패: {e}")
#             continue

#     # 상위 N개 정렬 반환
#     ranked = sorted(country_scores, key=lambda x: x[1], reverse=True)
#     return ranked[:top_n]

# recommend/recommend_countries.py
#%%
import os
import sys
sys.path.append('/Users/chaeyoung/Downloads/rag_project/recommend')
import pickle
from typing import List, Tuple
from scorer import calculate_score

VECTOR_DB_PATH = "vector_db/hs_based"

# 보기 좋게 표시할 국가명 매핑
COUNTRY_NAME_MAP = {
    "japan": "일본",
    "usa": "미국",
    "china": "중국",
    "vietnam": "베트남",
    "germany": "독일",
    "france": "프랑스",
    "taiwan": "대만",
    "russia": "러시아",
    "brazil": "브라질",
    "india": "인도",
    "mexico": "멕시코",
    "canada": "캐나다",
    "romania": "루마니아",
    "mongolia": "몽골",
    "thailand": "태국",
    "spain": "스페인",
    "argentina": "아르헨티나",
    "portugal": "포르투갈",
    "hungary": "헝가리",
    # 필요시 추가
}


def recommend_top_countries(hs_code: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """
    주어진 HS 코드에 대해 국가별 평균 점수를 계산하고, 상위 Top-N 국가를 반환합니다.

    Returns:
        List of tuples: [(country_name, score), ...]
    """
    
    country_scores = []
    

    for country in os.listdir(VECTOR_DB_PATH):
        path = os.path.join(VECTOR_DB_PATH, country, hs_code, "chunks.pkl")
        if not os.path.exists(path):
            continue

        try:
            with open(path, "rb") as f:
                chunks = pickle.load(f)

            total_score = 0
            valid_chunks = 0

            for chunk in chunks:
                try:
                    score = calculate_score(
                        import_val=chunk.get("import_val", 0.0),
                        tariff=chunk.get("tariff", 0.3),
                        tbt_level=chunk.get("tbt_level", 0.5)
                    )
                    total_score += score
                    valid_chunks += 1
                except Exception:
                    continue

            if valid_chunks > 0:
                avg_score = round(total_score / valid_chunks, 4)
                readable_name = COUNTRY_NAME_MAP.get(country, country)
                country_scores.append((readable_name, avg_score))

        except Exception as e:
            print(f"[!] {country} 처리 실패: {e}")

    # 상위 N개 국가 반환
    ranked = sorted(country_scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
# %%
