#%%
def calculate_score(import_val: float, tariff: float, tbt_level: float) -> float:
    """
    관세율, TBT 복잡도, 수입액 값을 기반으로 통합 점수를 계산합니다.
    점수는 0 ~ 1 사이 값으로 정규화됩니다.
    
    Args:
        import_val (float): 국가의 수입액 정규화 수치 (0~1, 높을수록 좋음)
        tariff (float): 관세율 (0~1, 낮을수록 좋음)
        tbt_level (float): TBT 복잡도 (0~1, 낮을수록 좋음)
    
    Returns:
        float: 종합 점수
    """

    # 가중치 (100점 기준)
    weight_import = 0.20
    weight_tariff = 0.45
    weight_tbt = 0.35

    # 관세율과 TBT는 낮을수록 좋으므로 역가중
    tariff_score = 1.0 - tariff
    tbt_score = 1.0 - tbt_level

    # 총합 점수 계산
    score = (
        weight_import * import_val +
        weight_tariff * tariff_score +
        weight_tbt * tbt_score
    )

    return round(score, 4)

# %%
