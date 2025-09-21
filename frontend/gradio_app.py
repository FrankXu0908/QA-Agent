import gradio as gr
import requests

# RAG FastAPI 地址
QA_API = "http://127.0.0.1:8000/qa"

def ask_qa(question):
    """
    调用 FastAPI QA 接口返回答案和来源
    """
    if not question.strip():
        return "请输入问题！", ""
    
    try:
        resp = requests.post(QA_API, json={"question": question}).json()
        answer = ""
        if isinstance(resp, dict):
            choices = resp.get("answer", {}).get("choices",[])  # 如果 FastAPI 直接返回模型原始输出
            if choices and len(choices) > 0:
                answer = choices[0].get("text", "")
            else:
                # 兼容老版本，尝试直接拿 answer 字段
                answer = resp.get("answer", "")
        else:
            answer = str(resp)
        sources = resp.get("sources", [])
        # 格式化来源
        sources_text = "\n".join([f"{i+1}. {hit['meta'].get('source','未知')} - 段落ID: {hit['meta'].get('id','N/A')}" 
                                  for i, hit in enumerate(sources)])
        return answer, sources_text
    except Exception as e:
        return f"调用接口出错: {e}", ""

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 质量文档 QA 系统（FMEA/PPAP 等）")
    question_input = gr.Textbox(label="请输入你的问题", lines=2, placeholder="例如：FMEA 的主要步骤是什么？")
    answer_output = gr.Textbox(label="答案", lines=8)
    sources_output = gr.Textbox(label="引用来源", lines=6)
    submit_btn = gr.Button("提问")

    submit_btn.click(fn=ask_qa, inputs=question_input, outputs=[answer_output, sources_output])

# 启动
if __name__ == "__main__":    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    # answer, sources_text = ask_qa("点检的流程是什么？")
    # print("Answer:", answer)