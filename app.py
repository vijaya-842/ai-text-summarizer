import streamlit as st
import tempfile
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2

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

# ── Get API Key ───────────────────────────────────────────────────────────────
def get_groq_key():
    import os
    key = os.environ.get("GROQ_API_KEY", "")
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("GROQ_API_KEY", "")
    return key

# ── Summarization functions ───────────────────────────────────────────────────
def stuff_summarize(text: str, model: str, tone: str, length: str) -> str:
    api_key = get_groq_key()
    llm = ChatGroq(model=model, temperature=0.3, api_key=api_key)

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
    api_key = get_groq_key()
    llm = ChatGroq(model=model, temperature=0.3, api_key=api_key)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

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


def load_document(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    else:
        return uploaded_file.read().decode("utf-8")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📝 Text Summarization")
    st.caption("Summarize any text or document!")
    st.divider()

    model = st.selectbox("🤖 Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
    ])

    st.divider()
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
    st.subheader("📁 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        if st.button("📥 Load Document", use_container_width=True):
            with st.spinner("Loading..."):
                st.session_state.original_text = load_document(uploaded)
                st.session_state.summary = ""
                st.success(f"✅ Loaded! ({len(st.session_state.original_text.split())} words)")

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.summary = ""
        st.session_state.original_text = ""
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📝 Text Summarization")
st.caption("Paste text or upload a document — get a smart summary!")

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
            preview = st.session_state.original_text[:2000]
            st.text(preview + "..." if len(st.session_state.original_text) > 2000 else preview)

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

    if st.session_state.original_text:
        col1, col2, col3 = st.columns(3)
        original_words = len(st.session_state.original_text.split())
        summary_words = len(st.session_state.summary.split())
        reduction = round((1 - summary_words / original_words) * 100)
        col1.metric("Original", f"{original_words} words")
        col2.metric("Summary", f"{summary_words} words")
        col3.metric("Reduced by", f"{reduction}%")

    st.download_button(
        label="⬇️ Download Summary",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain",
    )