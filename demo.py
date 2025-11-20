from dotenv import load_dotenv
import os

load_dotenv()  # 讀取 .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from mem0 import Memory
from openai import OpenAI

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize memory
memory = Memory()

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # 搜尋相關記憶
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

    # 建構系統提示，注入記憶
    system_prompt = f"You are a helpful AI. Answer based on query and memories.\nUser Memories:\n{memories_str}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    # 生成回應
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # 或其他模型
        messages=messages
    )
    assistant_response = response.choices[0].message.content

    # 儲存新對話為記憶
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response

# 運行聊天迴圈
if __name__ == "__main__":
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")
