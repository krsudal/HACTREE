# %%
# app_gradio_full.py

# import gradio as gr
# from ask_polyglot import answer_question_from_vectorstore
# from hs_predict.infer import predict_hs_code
# from tariff_barrier import answer_tariff_barrier_question
# from recommend.recommend_countries import recommend_top_countries

# # HS 코드 예측 함수
# def run_hs_prediction(item_name):
#     if not item_name.strip():
#         return "품목을 입력해주세요.", []
    
#     hs_code = predict_hs_code(
#         text=item_name,
#         model_path="hs_predict/hs_model.pt",
#         vocab_path="hs_predict/vocab.json",
#         label_encoder_path="hs_predict/label_encoder.pkl"
#     )

#     # 국가 추천
#     recommended = recommend_top_countries(hs_code, top_n=3)
#     country_list = [f"{name} (점수: {score:.3f})" for name, score in recommended]

#     return hs_code, country_list

# # 선택된 국가에 대한 전략 질의응답
# def respond_selected_country(question, hs_code, selected_country):
#     if not question.strip():
#         return "질문을 입력해주세요."
#     if not hs_code or not selected_country:
#         return "HS 코드와 국가 선택이 필요합니다."
    
#     country = selected_country.split(" ")[0]  # "Japan (점수: 0.785)" → "Japan"
#     return answer_tariff_barrier_question(hs_code, country, question)

# # Gradio UI 정의
# with gr.Blocks(title="HACTREE") as demo:
#     gr.Markdown("## HACTREE: HS code Auto Classification & Trade Report for Export Enterprise")

#     with gr.Tab("수출 전략 질문 응답 (Q&A)"):
#         question_input = gr.Textbox(
#             label="질문을 입력하세요",
#             placeholder="예: 일본에 화장품 수출 시 유의사항은?",
#             lines=2
#         )
#         answer_output = gr.Textbox(label="답변")
#         gr.Button("질문하기").click(fn=answer_question_from_vectorstore, inputs=question_input, outputs=answer_output)

#     with gr.Tab("HS 코드 기반 추천 및 전략 질의"):
#         with gr.Row():
#             item_input = gr.Textbox(
#                 label="제품명을 입력하세요",
#                 placeholder="예: 화장품, 전자레인지 등"
#             )
#             predict_button = gr.Button("HS 코드 예측 및 국가 추천")

#         hs_output = gr.Textbox(label="예측된 HS 코드")
#         country_dropdown = gr.Dropdown(label="추천 국가 선택", choices=[], interactive=True)
#         country_listbox = gr.Textbox(label="추천 국가 리스트", visible=False)  # 디버깅용

#         predict_button.click(
#             fn=run_hs_prediction,
#             inputs=item_input,
#             outputs=[hs_output, country_dropdown]
#         )

#         gr.Markdown("### 선택한 국가에 대해 전략 질문을 해보세요")
#         strategy_question = gr.Textbox(label="전략 질문", placeholder="예: 이 나라에서 인증 절차는 어떻게 되나요?", lines=2)
#         strategy_answer = gr.Textbox(label="AI 전략 요약 답변")

#         gr.Button("전략 질문하기").click(
#             fn=respond_selected_country,
#             inputs=[strategy_question, hs_output, country_dropdown],
#             outputs=strategy_answer
#         )

#     with gr.Tab("관세 및 TBT 질문 (기존)"):
#         tariff_input = gr.Textbox(
#             label="관세/TBT 관련 질문",
#             placeholder="예: 일본에서 화장품 수입 시 통관 절차는?"
#         )
#         tariff_output = gr.Textbox(label="답변")
#         gr.Button("질문하기").click(fn=answer_tariff_barrier_question, inputs=tariff_input, outputs=tariff_output)

# # 앱 실행
# if __name__ == "__main__":
#     demo.launch(share=True)

# %%
import gradio as gr
from ask_polyglot import answer_question_from_vectorstore
from hs_predict.infer import predict_hs_code
from tariff_barrier import answer_tariff_barrier_question
from recommend.recommend_countries import recommend_top_countries

# HS 코드 예측 + 국가 추천
def run_hs_prediction(item_name):
    if not item_name.strip():
        return "Error", gr.update(choices=[])

    hs_code = predict_hs_code(
        text=item_name,
        model_path="hs_predict/hs_model.pt",
        vocab_path="hs_predict/vocab.json",
        label_encoder_path="hs_predict/label_encoder.pkl"
    )

    try:
        recommended = recommend_top_countries(hs_code, top_n=3)
        # 👉 추천 결과를 문자열로 변환
        country_list = [f"{name} (점수: {score:.3f})" for name, score in recommended]
    except Exception as e:
        print(f"[!] 추천 실패: {e}")
        return hs_code, gr.update(choices=[])

    return hs_code, gr.update(choices=country_list, value=country_list[0] if country_list else None)


# 전략 질문 응답
def respond_selected_country(question, hs_code, selected_country):
    if not question.strip():
        return "질문을 입력해주세요."
    if not hs_code or not selected_country:
        return "HS 코드와 국가 선택이 필요합니다."
    
    country = selected_country.split(" ")[0]
    return answer_tariff_barrier_question(hs_code, country, question)

# Gradio 앱
with gr.Blocks(title="HACTREE") as demo:
    gr.Markdown("## HACTREE")

    with gr.Tab("수출 유망국 추천 및 전략 분석"):
        gr.Markdown("### 수출 품목을 입력하세요")

        with gr.Row():
            item_input = gr.Textbox(label="제품명", placeholder="예: 전자레인지", scale=3)
            predict_btn = gr.Button("HS 코드 예측 및 국가 추천", scale=1)

        hs_output = gr.Textbox(label="예측된 HS 코드", interactive=False)
        country_dropdown = gr.Dropdown(label="추천 수출 국가", choices=[], interactive=True)

        predict_btn.click(fn=run_hs_prediction, inputs=item_input, outputs=[hs_output, country_dropdown])

        gr.Markdown("### 선택한 국가에 대해 전략 질문을 해보세요")

        strategy_question = gr.Textbox(label="전략 질문", placeholder="예: 이 나라에서 인증 절차는 어떻게 되나요?", lines=2)
        strategy_answer = gr.Textbox(label="답변", lines=6)

        gr.Button("전략 질문하기").click(
            fn=respond_selected_country,
            inputs=[strategy_question, hs_output, country_dropdown],
            outputs=strategy_answer
        )

    with gr.Tab("일반 질의응답(Q&A)"):
        question_input = gr.Textbox(label="전략 질문", placeholder="예: 미국 시장 진출 전략은?")
        answer_output = gr.Textbox(label="답변", lines=6)
        gr.Button("질문하기").click(fn=answer_question_from_vectorstore, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()
# %%
