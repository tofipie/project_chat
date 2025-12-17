import streamlit as st
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

#from langchain_openai import ChatOpenAI

# --- 1. הגדרות מבנה הנתונים (מחוץ ללולאת הריצה) ---
class AgentState(TypedDict):
    project_name: Optional[str]
    data_type: Optional[str]
    budget: Optional[str]
    next_question: Optional[str]
    is_complete: bool
    user_input: str

class ExtractedInfo(BaseModel):
    project_name: Optional[str] = Field(None)
    data_type: Optional[str] = Field(None)
    budget: Optional[str] = Field(None)

# --- 2. לוגיקת הגרף (Nodes) ---
def extractor_node(state: AgentState):
  #  llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"]).with_structured_output(ExtractedInfo)
    llm  = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5).with_structured_output(ExtractedInfo)
    res = llm.invoke(f"Current State: {state}, User Input: {state['user_input']}")
    return {
        "project_name": res.project_name or state.get("project_name"),
        "data_type": res.data_type or state.get("data_type"),
        "budget": res.budget or state.get("budget"),
    }

def asker_node(state: AgentState):
    missing = []
    if not state.get("project_name"): missing.append("שם הפרויקט")
    if not state.get("data_type"): missing.append("סוג הנתונים (CSV, SQL וכו')")
    if not state.get("budget"): missing.append("תקציב מוערך")
    
    if missing:
        return {"is_complete": True, "next_question": "מעולה! אספתי את כל המידע הנדרש. תודה רבה."}
    
    #llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"])
    llm  = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

    question = llm.invoke(f"שאל שאלה קצרה רק על: {missing[0]}. מידע קיים: {state}").content
    return {"next_question": question, "is_complete": False}

# בניית הגרף (Compiled Graph)
workflow = StateGraph(AgentState)
workflow.add_node("extractor", extractor_node)
workflow.add_node("asker", asker_node)
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "asker")
workflow.add_edge("asker", END)
agent_app = workflow.compile()

# --- 3. ממשק Streamlit ---
st.set_page_config(page_title="AI Project Onboarding", page_icon="🤖")
st.title("🤖 סוכן אפיון פרויקטים")
st.markdown("הסוכן ישאל אותך שאלות עד שכל פרטי הפרויקט יהיו מלאים.")

# אתחול ה-Session State
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "project_name": None, "data_type": None, "budget": None,
        "next_question": "היי! בוא נתחיל. איך היית רוצה לקרוא לפרויקט שלך?",
        "is_complete": False, "user_input": ""
    }
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("assistant", st.session_state.agent_state["next_question"])]

# הצגת היסטוריית הצ'אט
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# תיבת קלט למשתמש
if not st.session_state.agent_state["is_complete"]:
    if user_prompt := st.chat_input("הקלד את תשובתך כאן..."):
        # 1. הצגת הודעת המשתמש
        st.session_state.chat_history.append(("user", user_prompt))
        with st.chat_message("user"):
            st.write(user_prompt)
        
        # 2. הרצת הגרף עם הקלט החדש
        st.session_state.agent_state["user_input"] = user_prompt
        new_state = agent_app.invoke(st.session_state.agent_state)
        
        # 3. עדכון ה-State וההיסטוריה
        st.session_state.agent_state.update(new_state)
        st.session_state.chat_history.append(("assistant", st.session_state.agent_state["next_question"]))
        
        # 4. ריענון הממשק להצגת תגובת הסוכן
        st.rerun()

# הצגת המידע שנאסף בצד (Sidebar)
with st.sidebar:
    st.header("📊 מידע שנאסף")
    st.write(f"**שם הפרויקט:** {st.session_state.agent_state['project_name'] or '---'}")
    st.write(f"**סוג דאטה:** {st.session_state.agent_state['data_type'] or '---'}")
    st.write(f"**תקציב:** {st.session_state.agent_state['budget'] or '---'}")
    if st.session_state.agent_state["is_complete"]:
        st.success("✅ האפיון הושלם!")
