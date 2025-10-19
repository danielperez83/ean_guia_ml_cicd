import gradio as gr

def greet(name):
    return f"Hola {name}! 🎈"

demo = gr.Interface(fn=greet, inputs="text", outputs="text", title="Drug-Classification (demo)")
demo.launch()
