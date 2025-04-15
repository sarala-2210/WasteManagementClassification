import gradio as gr
import os
demo = gr.load("sarala-2210/NewsArticleAndLinkedInPostGenerator", src="models", hf_token=os.environ.get("HF_TOKEN"), examples=None).launch(ssr_mode=False)
