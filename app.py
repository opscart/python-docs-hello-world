from flask import Flask, request, jsonify
import pyautogen as autogen
import os

app = Flask(__name__)

# ✅ Read API key from env
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your environment variables.")

# ✅ Set up AutoGen config
# This assumes you have a file `OAI_CONFIG_LIST.json` in the repo root
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={"model": ["gpt-4", "gpt-3.5-turbo"]}
)

# ✅ Create the AssistantAgent
assistant = autogen.AssistantAgent(
    name="AutoGenAssistant",
    llm_config={
        "config_list": config_list,
        "api_key": openai_api_key
    }
)

# ✅ Setup UserProxyAgent to simulate user interaction
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False
)

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message")
    messages = [{"role": "user", "content": user_input}]

    # ✅ Let user_proxy initiate a chat with the assistant
    user_proxy.initiate_chat(
        assistant,
        message=user_input
    )

    # ✅ Extract latest message sent by assistant
    last_message = assistant.chat_messages[-1]["content"]

    return jsonify({"response": last_message})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
