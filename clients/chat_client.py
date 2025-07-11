import streamlit as st
import grpc
import json
from generated import generic_service_pb2, generic_service_pb2_grpc

# Setup gRPC channel and stub
channel = grpc.insecure_channel("localhost:50051")
stub = generic_service_pb2_grpc.GenericServiceStub(channel)

# Streamlit page config
st.set_page_config(page_title="gRPC Chatbot", page_icon="💬", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 AI Chatbot</h1>", unsafe_allow_html=True)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    query_id = st.text_input("Query ID", "demo-session-2")
    source = st.text_input("Source", "movie")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_new_tokens = st.slider("Max Tokens", 32, 1024, 256)
    clear = st.button("🧹 Clear Chat")
    if clear:
        st.session_state.chat_history = []

# Display chat history with latest on top and colored text
for entry in reversed(st.session_state.chat_history):
    if entry["role"] == "user":
        st.markdown(f"""
            <div style="
                background-color:#DCF8C6;
                color:#000080;  /* Navy blue text */
                padding:10px;
                border-radius:10px;
                margin:10px 0;
                text-align:right;
                font-weight:bold;
                font-family: Arial, sans-serif;
            ">
                You: {entry['content']}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="
                background-color:#F1F0F0;
                color:#800000; /* Maroon text */
                padding:10px;
                border-radius:10px;
                margin:10px 0;
                text-align:left;
                font-weight:bold;
                font-family: Arial, sans-serif;
            ">
                Bot: {entry['content']}
            </div>
        """, unsafe_allow_html=True)

# Input form at bottom
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", height=80)
    submitted = st.form_submit_button("Send")

# When user submits message
if submitted and user_input.strip():
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Prepare request payload
    payload = {
        "QueryId": query_id,
        "Query": [user_input],
        "Source": source,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "clearchat": clear
    }

    request = generic_service_pb2.RpcRequest(
        RpcContext="Chat Client",
        MethodName="chat",
        MethodParamType="json",
        MethodParamData=json.dumps(payload)
    )

    try:
        response = stub.ExecuteRemoteMethod(request)
        response_data = json.loads(response.ResponseData)
        bot_reply = response_data.get("Response", "")

        # Append bot's reply
        if bot_reply:
            st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
        else:
            st.session_state.chat_history.append({"role": "bot", "content": " [No reply received]"})

    except grpc.RpcError as e:
        st.session_state.chat_history.append({
            "role": "bot",
            "content": f"❌ gRPC Error: {e.code()} - {e.details()}"
        })

# Debug info (optional)
with st.expander("🛠 Debug Info"):
    st.write("Chat History:", st.session_state.chat_history)

