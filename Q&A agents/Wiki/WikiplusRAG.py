import streamlit as st
from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
from langchain.agents import create_tool_calling_agent
from langchain.agents import initialize_agent, AgentType

# --- UI Setup ---
st.set_page_config(page_title="KRCL RuleBot", page_icon="Konkan_Railway_logo.svg.png")
st.image("Konkan_Railway_logo.svg.png", width=50)
st.title("📘 KRCL G&SR 2020 Assistant")

# --- Load Embeddings & Vector DB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index_krcl", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# --- Define Wiki Tool ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- Define Custom Retrieval Function ---
def krcl_retriever_tool(query: str) -> str:
    """Retrieve rule-based context from FAISS for a given query."""
    results = retriever.get_relevant_documents(query)
    if not results:
        return "The information is not available in the provided rules."
    return "\n\n".join([doc.page_content for doc in results])

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
# KRCL Rules Expert System v3.1
**Railway Safety-Critical Knowledge Assistant**

## Operational Mandate
⚠️ **All responses must maintain railway operational safety** ⚠️
- Errors could lead to accidents or disciplinary actions
- Maximum precision required at all times
- When uncertain: "This is not specified in KRCL rules"

## Knowledge Protocol
### Primary Source (Mandatory):
1. **KRCL General and Subsidiary Rules 2020**
   - Always begin with exact rule references
   - Use verbatim rule wording
   - Include full rule identifiers (e.g., GR 5.01(3), SR 2.14)

### Secondary Source (Restricted):
2. **Wikipedia/Public Sources** (Only when):
   a) Query is definition/terminology based
   b) No KRCL rule exists after thorough search
   c) Information has no operational impact
   d) Must include:
      "🔍 [External Reference]"  
      "This general information is not in KRCL rules but according to public sources..."

## Response Structure
### 1. Operational Answer
[Clear, concise response to the query]

### 2. Rule Evidence
Format rules EXACTLY as follows:
#### [Rule Number] [Rule Title]
• (1) [Exact wording from manual]
• (2) [Exact wording]
  - (a) [Sub-clause if present]

[Separate rules with '---' if multiple]

### 3. External Reference (If Absolutely Required)
🔍 [External Source Notice]
• [Brief factual statement]
• "KRCL rules govern actual implementation"

## Prohibitions
❌ Never mix rule content with external knowledge
❌ No operational procedures from external sources
❌ No paraphrasing of safety-critical rules

## Special Cases
• Definitions: "As defined in GR X.Y: [exact text]"
• Procedures: "Per SR X.Y(2): [step-by-step]"
• Missing info: "Not specified in KRCL rules"
• Contradictions: "Rule GR X.Y takes precedence"
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("user", "{input}")
])
# --- LLM & Tools Setup ---
llm = Ollama(model="mistral")  # Adjust model name as per your setup
tools = [
    Tool(
        name="KRCLRulesRetriever",
        func=krcl_retriever_tool,
        description="Retrieves accurate rule-based answers from the KRCL G&SR 2020 manual."
    ),
    wiki
]

# --- Agent Creation ---
#agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # or AgentType.CONVERSATIONAL_REACT_DESCRIPTION
    verbose=True
)

#-----rule format-------

def format_rules(text: str) -> str:
    # Normalize text
    text = text.replace("S.R.", "SR").replace("G.R.", "GR")
    
    # Split by rule headers (GR, SR, or numbered rules like 5.01 etc.)
    rule_sections = re.split(r"\n*(?=(?:GR|SR|Rule)?\s?\d{1,2}\.\d{1,2})", text.strip())

    formatted = ""
    for section in rule_sections:
        lines = section.strip().split("\n")
        if not lines or len(lines[0].strip()) < 3:
            continue  # skip empty or bad sections

        # First line is Rule Title
        rule_title = lines[0].strip()
        formatted += f"### 📘 {rule_title}\n"

        # Remaining lines are clauses
        for line in lines[1:]:
            line = line.strip()
            if re.match(r"^\(\d+\)", line):  # clause like (1), (2)
                formatted += f"- {line}\n"
            elif re.match(r"^\s*-\s*\([a-z]\)", line, re.I):  # sub-clause like -(a)
                formatted += f"  - {line}\n"
            elif line:
                formatted += f"  - {line}\n"
        
        formatted += "---\n"
    return formatted


# --- Query Input UI ---
# --- Query Input UI ---
query = st.text_input("Enter your query about KRCL Rules ")

if query:
    with st.spinner(" Thinking..."):
        try:
            response = agent_executor.invoke({
                "input": query,
                "chat_history": []  #  Required variable
            })
            output = response.get("output", "")

            # Split final answer and rule evidence
            rule_split = re.split(r"(GR|SR|Rule)\s", output, maxsplit=1)
            if len(rule_split) > 1:
                final_answer = rule_split[0].strip()
                rule_evidence = " ".join(rule_split[1:]).strip()
            else:
                final_answer = output.strip()
                rule_evidence = "No direct rule citation found."

            # --- Display Output ---
            st.subheader(" Final Answer")
            st.markdown(final_answer)

            st.subheader(" Rule Observation")
            st.markdown(format_rules(rule_evidence))

        except Exception as e:
            st.error(f"Something went wrong: {e}")
