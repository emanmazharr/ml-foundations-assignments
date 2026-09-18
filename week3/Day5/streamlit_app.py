import streamlit as st
from afl_langgraph_app import ask
st.set_page_config(page_title='AFL Assistant')
st.title('🏉 AFL Assistant')
st.caption('AFL-only • dataset-grounded • prediction disclaimer included')
cid=st.session_state.setdefault('cid','streamlit-demo')
for role,msg in st.session_state.setdefault('messages',[]): st.chat_message(role).write(msg)
if q:=st.chat_input('Ask an AFL question...'):
    st.session_state.messages.append(('user',q)); st.chat_message('user').write(q)
    r=ask(q,cid); st.session_state.messages.append(('assistant',r['response'])); st.chat_message('assistant').write(r['response'])
    with st.expander('Monitoring / trace'): st.json(r)
