import streamlit as st
import tempfile
import os
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarization 📝",
    page_icon="📝",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "original_text" not in st.session_state:
    st.session_state.original_text = ""

# ── Summarization functions ───────────────────────────────────────────────────

def stuff_summarize(text: str, model: str, tone: str, length: str) -> str:
    """Stuff strategy — put everything in one prompt. Best for short texts."""
    llm = ChatOllama(model=model, temperature=0.3)

    length_map = {
        "Very Short (1-2 sentences)": "1-2 sentences",
        "Short (1 paragraph)": "1 short paragraph",
        "Medium (2-3 paragraphs)": "2-3 paragraphs",
        "Detailed (5+ paragraphs)": "5 or more paragraphs",
    }

    prompt = ChatPromptTemplate.from_template("""
You are an expert summarizer. Summarize the following text.

Tone: {tone}
Length: {length}

Text to summarize:
{text}

Summary:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "text": text,
        "tone": tone,
        "length": length_map[length],
    })


def map_reduce_summarize(text: str, model: str, tone: str, length: str) -> str:
    """Map-Reduce strategy — summarize chunks then combine. Best for long texts."""
    llm = ChatOllama(model=model, temperature=0.3)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)

    # MAP — summarize each chunk
    map_prompt = ChatPromptTemplate.from_template("""
Summarize this section of text in 2-3 sentences:

{chunk}

Summary:""")
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []
    progress = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = map_chain.invoke({"chunk": chunk})
        chunk_summaries.append(summary)
        progress.progress((i + 1) / len(chunks))

    # REDUCE — combine all chunk summaries
    combined = "\n\n".join(chunk_summaries)

    length_map = {
        "Very Short (1-2 sentences)": "1-2 sentences",
        "Short (1 paragraph)": "1 short paragraph",
        "Medium (2-3 paragraphs)": "2-3 paragraphs",
        "Detailed (5+ paragraphs)": "5 or more paragraphs",
    }

    reduce_prompt = ChatPromptTemplate.from_template("""
You are an expert summarizer. Combine these section summaries into one final summary.

Tone: {tone}
Final Length: {length}

Section Summaries:
{summaries}

Final Summary:""")
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    return reduce_chain.invoke({
        "summaries": combined,
        "tone": tone,
        "length": length_map[length],
    })


def load_pdf(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    docs = loader.load()
    os.unlink(tmp_path)
    return "\n\n".join(doc.page_content for doc in docs)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📝 Text Summarization")
    st.caption("Summarize any text or document!")
    st.divider()

    # Model
    model = st.selectbox("🤖 Model", ["llama3.2", "mistral", "phi3"])

    st.divider()

    # Strategy
    st.subheader("📋 Strategy")
    strategy = st.radio(
        "Choose method:",
        ["Stuff (Fast)", "Map-Reduce (Large docs)"],
    )
    st.caption(
        "**Stuff** → best for short texts (< 2000 words)\n\n"
        "**Map-Reduce** → best for long texts/PDFs"
    )

    st.divider()

    # Output settings
    st.subheader("⚙️ Output Settings")
    tone = st.selectbox("Tone", [
        "Neutral and professional",
        "Simple and easy to understand",
        "Academic and formal",
        "Bullet points",
    ])
    length = st.selectbox("Summary Length", [
        "Very Short (1-2 sentences)",
        "Short (1 paragraph)",
        "Medium (2-3 paragraphs)",
        "Detailed (5+ paragraphs)",
    ])

    st.divider()

    # Upload PDF
    st.subheader("📁 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        if st.button("📥 Load Document", use_container_width=True):
            with st.spinner("Loading..."):
                st.session_state.original_text = load_pdf(uploaded)
                st.session_state.summary = ""
            st.success(f"✅ Loaded! ({len(st.session_state.original_text.split())} words)")

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.summary = ""
        st.session_state.original_text = ""
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📝 Text Summarization")
st.caption("Paste text or upload a document — get a smart summary!")

# Two tabs
tab1, tab2 = st.tabs(["✍️ Paste Text", "📄 Uploaded Document"])

with tab1:
    text_input = st.text_area(
        "Paste your text here:",
        height=300,
        placeholder="Paste any article, essay, report, or text here...",
    )

    if st.button("🚀 Summarize Text", use_container_width=True, type="primary"):
        if not text_input.strip():
            st.warning("Please paste some text first!")
        else:
            with st.spinner("Summarizing..."):
                try:
                    if "Map-Reduce" in strategy:
                        summary = map_reduce_summarize(text_input, model, tone, length)
                    else:
                        summary = stuff_summarize(text_input, model, tone, length)
                    st.session_state.summary = summary
                    st.session_state.original_text = text_input
                except Exception as e:
                    st.error(f"❌ Error: {e}")

with tab2:
    if st.session_state.original_text:
        word_count = len(st.session_state.original_text.split())
        st.info(f"📄 Document loaded: **{word_count} words**")

        with st.expander("👀 Preview Document"):
            st.text(st.session_state.original_text[:2000] + "..." if len(st.session_state.original_text) > 2000 else st.session_state.original_text)

        if st.button("🚀 Summarize Document", use_container_width=True, type="primary"):
            with st.spinner("Summarizing document..."):
                try:
                    if "Map-Reduce" in strategy:
                        summary = map_reduce_summarize(
                            st.session_state.original_text, model, tone, length
                        )
                    else:
                        summary = stuff_summarize(
                            st.session_state.original_text, model, tone, length
                        )
                    st.session_state.summary = summary
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.info("👈 Upload a PDF or TXT file from the sidebar first!")

# ── Show Summary ──────────────────────────────────────────────────────────────
if st.session_state.summary:
    st.divider()
    st.subheader("✅ Summary")
    st.markdown(st.session_state.summary)

    # Stats
    if st.session_state.original_text:
        col1, col2, col3 = st.columns(3)
        original_words = len(st.session_state.original_text.split())
        summary_words = len(st.session_state.summary.split())
        reduction = round((1 - summary_words/original_words) * 100)

        col1.metric("Original", f"{original_words} words")
        col2.metric("Summary", f"{summary_words} words")
        col3.metric("Reduced by", f"{reduction}%")

    # Download button
    st.download_button(
        label="⬇️ Download Summary",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain",
    )